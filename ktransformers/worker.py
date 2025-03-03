import torch
import zmq
import pickle
from typing import Dict, Any
import logging
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, GenerationConfig
import time
from transformers import TextStreamer
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model_and_tokenizer(model_path: str, gguf_path: str):
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    logger.info("Loading model config...")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    torch.set_default_dtype(config.torch_dtype)
    
    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        trust_remote_code=True,
        device_map="auto"
    )
    model.eval()
    
    return model, tokenizer

class ModelWorker:
    def __init__(self, address: str):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        host, port = address.split(':')
        self.socket.bind(f"tcp://0.0.0.0:{port}")
        self.socket.setsockopt(zmq.RCVHWM, 0)
        self.socket.setsockopt(zmq.SNDHWM, 0)
        
        # 设置 CPU 核心数
        torch.set_num_threads(10)  # 限制为10个核心
        logger.info(f"Set PyTorch CPU threads to 10")
        
        self.model = None
        self.tokenizer = None
        logger.info(f"Worker started at {address}, listening on all interfaces")

    def load_model(self, model_path: str, gguf_path: str):
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        logger.info("Loading model config...")
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        torch.set_default_dtype(config.torch_dtype)
        
        # 获取可用的 GPU
        gpu_memory = {}
        for i in range(torch.cuda.device_count()):
            total_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            # P100 有 16GB 显存，我们保留一些用于中间计算
            if torch.cuda.get_device_name(i).find("P100") != -1:
                gpu_memory[i] = f"{int(total_mem * 0.8)}GiB"  # 使用 80% 的显存
        
        logger.info(f"Available GPUs: {gpu_memory}")
        
        # 添加 CPU 内存配置
        max_memory = {
            **gpu_memory,  # GPU 显存配置
            'cpu': "138GiB"  # CPU 内存配置
        }
        
        logger.info(f"Memory configuration: {max_memory}")
        
        # 加载模型时自动分配到多个 GPU
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            trust_remote_code=True,
            device_map="auto",  # 自动决定最优的设备分配
            max_memory=max_memory,
            torch_dtype=torch.float16,
            offload_folder=None,
            offload_state_dict=False
        )
        self.model.eval()
        
        # 记录模型分布情况
        if hasattr(self.model, 'hf_device_map'):
            logger.info("Model layer distribution:")
            for layer, device in self.model.hf_device_map.items():
                logger.info(f"  {layer}: {device}")
        
        # 设置生成配置
        try:
            self.model.generation_config = GenerationConfig.from_pretrained(model_path)
        except Exception as e:
            logger.warning(f"Generation config can't auto create, making default. Message: {e}")
            self.model.generation_config = GenerationConfig(
                temperature=0.6,
                top_p=0.9,
                do_sample=True
            )
        if self.model.generation_config.pad_token_id is None:
            self.model.generation_config.pad_token_id = self.model.generation_config.eos_token_id
            
        logger.info("Model loaded successfully")

    def generate(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            logger.info("Received generation request")
            logger.info(f"Input sequence length: {len(input_data['input_ids'])}")
            
            # 创建输入张量
            logger.info("Moving input tensors to GPU...")
            input_ids = torch.tensor(input_data['input_ids'], device='cuda')
            attention_mask = torch.tensor(input_data['attention_mask'], device='cuda')
            
            # 生成参数日志
            max_new_tokens = input_data.get('max_new_tokens', 512)
            temperature = input_data.get('temperature', 0.6)
            top_p = input_data.get('top_p', 0.9)
            logger.info(f"Generation parameters: max_new_tokens={max_new_tokens}, "
                       f"temperature={temperature}, "
                       f"top_p={top_p}")
            
            # 生成文本
            logger.info("Starting text generation...")
            start_time = time.time()
            with torch.no_grad(), torch.amp.autocast('cuda'):
                logger.info("Initializing generation process...")
                # 添加详细的初始化信息
                logger.info("- Checking model state...")
                logger.info(f"- Model device: {self.model.device}")
                logger.info(f"- Input shape: {input_ids.shape}")
                logger.info(f"- Current GPU memory allocated: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
                
                # KV cache 信息
                num_layers = len(self.model.model.layers)
                hidden_size = self.model.config.hidden_size
                logger.info(f"- Model architecture:")
                logger.info(f"  - Number of layers: {num_layers}")
                logger.info(f"  - Hidden size: {hidden_size}")
                
                # 估算 KV cache 大小
                seq_len = input_ids.shape[1] + max_new_tokens
                kv_cache_size = 2 * num_layers * seq_len * hidden_size * input_ids.shape[0] * 2 / (1024**3)  # in GB
                logger.info(f"- Estimated KV cache size: {kv_cache_size:.2f}GB")
                
                # 检查是否启用了 flash attention
                if hasattr(self.model.config, '_attn_implementation'):
                    logger.info(f"- Attention implementation: {self.model.config._attn_implementation}")
                
                logger.info("- Starting token generation...")
                logger.info("  - Generation started with following settings:")
                logger.info(f"    - Input context length: {len(input_ids[0])} tokens")
                logger.info(f"    - Maximum new tokens: {max_new_tokens}")
                logger.info(f"    - Temperature: {temperature}")
                logger.info(f"    - Top-p sampling: {top_p}")
                logger.info(f"    - Do sample: {input_data.get('do_sample', True)}")
                
                # 修改 streamer 配置
                class CustomStreamer(TextStreamer):
                    def on_finalized_text(self, text: str, stream_end: bool = False):
                        logger.info(f"    - Generated text: {text}")
                
                streamer = CustomStreamer(
                    self.tokenizer, 
                    skip_prompt=True,
                    skip_special_tokens=True
                )
                
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=input_data.get('do_sample', True),
                    streamer=streamer
                )
                
                # 记录生成完成的信息
                final_time = time.time()
                total_gen_time = final_time - start_time
                logger.info("  - Generation completed:")
                logger.info(f"    - Total generation time: {total_gen_time:.2f} seconds")
                logger.info(f"    - Final sequence length: {len(outputs[0])} tokens")
                logger.info(f"    - New tokens generated: {len(outputs[0]) - len(input_ids[0])}")
                logger.info(f"    - Average speed: {(len(outputs[0]) - len(input_ids[0])) / total_gen_time:.2f} tokens/sec")
            
            end_time = time.time()
            generation_time = end_time - start_time
            tokens_generated = len(outputs[0]) - len(input_ids[0])
            tokens_per_second = tokens_generated / generation_time
            
            logger.info(f"Generation completed in {generation_time:.2f} seconds")
            logger.info(f"Tokens generated: {tokens_generated}")
            logger.info(f"Generation speed: {tokens_per_second:.2f} tokens/second")
            logger.info(f"Final sequence length: {len(outputs[0])}")
            
            # 移动到CPU并转换格式
            logger.info("Moving outputs to CPU and preparing response...")
            start_transfer = time.time()
            
            # 使用异步传输
            outputs_cpu = outputs.to('cpu', non_blocking=True)
            
            # 在等待GPU->CPU传输的同时，记录GPU内存使用情况
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"GPU Memory: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")
            
            # 确保传输完成
            torch.cuda.synchronize()
            
            # 使用 numpy() 的异步版本
            with torch.no_grad():
                output_ids = outputs_cpu.numpy(force=True)  # force=True 可能会更快
            
            transfer_time = time.time() - start_transfer
            logger.info(f"Data transfer and conversion completed in {transfer_time:.2f} seconds")
            
            return {
                'output_ids': output_ids.tolist(),
                'status': 'success'
            }
        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }

    def run(self):
        logger.info("Worker ready to receive requests")
        while True:
            try:
                message = self.socket.recv_pyobj()
                command = message.get('command')
                
                if command == 'load_model':
                    self.load_model(message['model_path'], message['gguf_path'])
                    self.socket.send_pyobj({'status': 'success'})
                
                elif command == 'generate':
                    result = self.generate(message['data'])
                    self.socket.send_pyobj(result)
                
                elif command == 'shutdown':
                    logger.info("Shutting down worker")
                    self.socket.send_pyobj({'status': 'shutdown'})
                    break
                
            except Exception as e:
                logger.error(f"Error processing message: {str(e)}")
                self.socket.send_pyobj({'status': 'error', 'error': str(e)})

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker_address', type=str, required=True)
    parser.add_argument('--worker_id', type=str, required=True,
                      help='Worker ID (e.g., "worker1" or "worker2")')
    args = parser.parse_args()
    
    # 加载配置
    worker_config = WORKER_CONFIGS[args.worker_id]
    shard_strategy = MODEL_SHARD_STRATEGY[args.worker_id]
    
    # 设置可见GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
        [gpu.split(":")[-1] for gpu in worker_config["gpus"]]
    )
    
    # 设置CPU线程
    torch.set_num_threads(worker_config["cpu_threads"])
    
    worker = ModelWorker(worker_config["address"])
    worker.run()

if __name__ == '__main__':
    main() 