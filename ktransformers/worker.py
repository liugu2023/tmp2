import torch
import zmq
import pickle
from typing import Dict, Any
import logging
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, GenerationConfig
import time
from transformers import TextStreamer

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
        
        self.model = None
        self.tokenizer = None
        logger.info(f"Worker started at {address}, listening on all interfaces")

    def load_model(self, model_path: str, gguf_path: str):
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        logger.info("Loading model config...")
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        torch.set_default_dtype(config.torch_dtype)
        
        logger.info("Loading model on CPU...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            trust_remote_code=True,
            device_map='cpu',  # 在CPU上加载
            torch_dtype=torch.float16,  # 使用半精度
            low_cpu_mem_usage=True,     # 降低CPU内存使用
        )
        self.model.eval()
        
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
        
        logger.info("Model loaded successfully on CPU")

    def generate(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            logger.info("Received generation request")
            logger.info(f"Input sequence length: {len(input_data['input_ids'])}")
            
            # 将模型移动到GPU
            logger.info("Moving model to GPU...")
            self.model = self.model.cuda()
            
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
                logger.info("  - Preparing generation config:")
                generation_config = self.model.generation_config
                logger.info(f"    - Repetition penalty: {generation_config.repetition_penalty}")
                logger.info(f"    - Length penalty: {generation_config.length_penalty}")
                logger.info(f"    - Early stopping: {generation_config.early_stopping}")
                logger.info(f"    - Pad token ID: {generation_config.pad_token_id}")
                logger.info(f"    - EOS token ID: {generation_config.eos_token_id}")
                
                # 记录起始时间戳
                gen_start_time = time.time()
                last_log_time = gen_start_time
                
                def progress_callback(step: int, token: int):
                    nonlocal last_log_time
                    current_time = time.time()
                    # 每2秒记录一次进度
                    if current_time - last_log_time >= 2:
                        tokens_so_far = step - len(input_ids[0])
                        time_elapsed = current_time - gen_start_time
                        speed = tokens_so_far / time_elapsed if time_elapsed > 0 else 0
                        logger.info(f"    - Generated {tokens_so_far} tokens, "
                                  f"speed: {speed:.2f} tokens/sec, "
                                  f"time elapsed: {time_elapsed:.2f}s")
                        last_log_time = current_time
                
                # 设置 streamer 的回调函数
                streamer = TextStreamer(
                    self.tokenizer, 
                    skip_prompt=True,
                    skip_special_tokens=True
                )
                streamer.on_finalized_text = lambda x: logger.info(f"    - Generated text: {x}")
                
                logger.info("  - Generation started with following settings:")
                logger.info(f"    - Input context length: {len(input_ids[0])} tokens")
                logger.info(f"    - Maximum new tokens: {max_new_tokens}")
                logger.info(f"    - Temperature: {temperature}")
                logger.info(f"    - Top-p sampling: {top_p}")
                logger.info(f"    - Do sample: {input_data.get('do_sample', True)}")
                
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=input_data.get('do_sample', True),
                    streamer=streamer
                )
            
            # 将模型移回CPU，释放GPU内存
            logger.info("Moving model back to CPU...")
            self.model = self.model.cpu()
            torch.cuda.empty_cache()
            
            # 移动输出到CPU
            logger.info("Moving outputs to CPU...")
            outputs_cpu = outputs.to('cpu', non_blocking=True)
            torch.cuda.synchronize()
            
            # 记录GPU内存使用情况
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"GPU Memory: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")
            
            logger.info("Preparing response...")
            return {
                'output_ids': outputs_cpu.numpy().tolist(),
                'status': 'success'
            }
        except Exception as e:
            # 确保在发生错误时也将模型移回CPU
            try:
                self.model = self.model.cpu()
                torch.cuda.empty_cache()
            except:
                pass
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
    args = parser.parse_args()
    
    worker = ModelWorker(args.worker_address)
    worker.run()

if __name__ == '__main__':
    main() 