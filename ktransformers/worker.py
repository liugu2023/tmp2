import torch
import zmq
import pickle
from typing import Dict, Any
import logging
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, GenerationConfig, TextStreamer
import time
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
        
        self.model = None
        self.tokenizer = None
        logger.info(f"Worker started at {address}, listening on all interfaces")

    def load_model(self, model_path: str, gguf_path: str):
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        logger.info("Loading model config...")
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        
        # 更新配置
        config.attn_implementation = "flash_attention_2"  # 新的 Flash Attention 2 配置
        torch.set_default_dtype(config.torch_dtype)
        
        # 初始化分布式环境
        if not torch.distributed.is_initialized():
            logger.info("Initializing distributed environment...")
            torch.distributed.init_process_group(
                backend='nccl',
                init_method='tcp://localhost:29501',
                world_size=1,
                rank=0
            )
        
        logger.info("Loading model...")
        # 设置双 GPU 的内存分配和并行配置
        max_memory = {
            0: "12GiB",    # 第一张 GPU 使用 12GB
            1: "12GiB",    # 第二张 GPU 使用 12GB
            'cpu': "138GiB" # 剩余放在 CPU 内存
        }
        
        # 设置 CPU 线程数为物理核心数
        num_physical_cores = 6
        num_threads = 12  # 总线程数
        
        # 设置 PyTorch 线程数
        torch.set_num_threads(num_threads)  # 使用所有逻辑核心
        
        # 设置 OpenMP 线程数
        os.environ["OMP_NUM_THREADS"] = str(num_physical_cores)  # OpenMP 使用物理核心数
        os.environ["MKL_NUM_THREADS"] = str(num_physical_cores)  # MKL 使用物理核心数
        
        logger.info("CPU threading configuration:")
        logger.info(f"- Physical cores: {num_physical_cores}")
        logger.info(f"- Total threads: {num_threads}")
        logger.info(f"- PyTorch threads: {torch.get_num_threads()}")
        logger.info(f"- OpenMP threads: {os.environ.get('OMP_NUM_THREADS')}")
        logger.info(f"- MKL threads: {os.environ.get('MKL_NUM_THREADS')}")
        
        # 添加并行配置
        model_kwargs = {
            "device_map": "auto",
            "max_memory": max_memory,
            "low_cpu_mem_usage": True,
        }
        
        # 设置并行环境
        if torch.cuda.is_available():
            # 启用 CUDA 优化
            torch.backends.cudnn.benchmark = True  # 启用 cuDNN 自动调优
            torch.backends.cudnn.enabled = True
            torch.backends.cuda.matmul.allow_tf32 = True  # 允许使用 TF32
            torch.backends.cudnn.allow_tf32 = True
        
        logger.info("Applying optimization configurations...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            **model_kwargs
        )
        
        # 应用模型优化
        if hasattr(self.model, "config"):
            self.model.config.use_cache = True  # 启用 KV 缓存
            if hasattr(self.model.config, "pretraining_tp"):
                self.model.config.pretraining_tp = 1  # 设置张量并行度
        
        # 输出优化配置信息
        logger.info("Model optimization settings:")
        logger.info("- Using FP16 precision")
        logger.info("- Flash Attention 2 enabled")
        logger.info("- CPU threads:")
        logger.info(f"- Physical cores: {num_physical_cores}")
        logger.info(f"- Total threads: {num_threads}")
        logger.info(f"- PyTorch threads: {torch.get_num_threads()}")
        logger.info(f"- OpenMP threads: {os.environ.get('OMP_NUM_THREADS')}")
        logger.info(f"- MKL threads: {os.environ.get('MKL_NUM_THREADS')}")
        logger.info("- Device mapping:")
        for name, device in self.model.hf_device_map.items():
            logger.info(f"  {name}: {device}")
        
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
            
        logger.info("Model loaded successfully")

    # 修改 CustomStreamer 类定义
    class CustomStreamer(TextStreamer):
        def __init__(self, tokenizer, on_text_chunk=None, **kwargs):
            super().__init__(tokenizer, **kwargs)
            self.on_text_chunk = on_text_chunk

        def on_finalized_text(self, text: str, stream_end: bool = False):
            if self.on_text_chunk:
                self.on_text_chunk(text, stream_end)

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
            
            # 创建自定义流式处理器
            class StreamingHandler:
                def __init__(self, socket, tokenizer):
                    self.socket = socket
                    self.tokenizer = tokenizer
                    self.current_text = ""
                    self.last_send_time = time.time()
                    
                def __call__(self, text_chunk: str, stream_end: bool = False):
                    current_time = time.time()
                    self.current_text += text_chunk
                    
                    # 每0.5秒或结束时发送一次
                    if stream_end or (current_time - self.last_send_time) > 0.5:
                        self.socket.send_pyobj({
                            'status': 'streaming',
                            'text': self.current_text,
                            'finished': stream_end
                        })
                        # 等待客户端确认
                        self.socket.recv_pyobj()
                        self.last_send_time = current_time
            
            # 创建流式处理器
            streamer = self.CustomStreamer(
                self.tokenizer,
                on_text_chunk=StreamingHandler(self.socket, self.tokenizer),
                skip_prompt=True,
                skip_special_tokens=True
            )
            
            # 生成文本
            logger.info("Starting text generation...")
            start_time = time.time()
            
            # 设置生成时的线程配置
            torch.set_grad_enabled(False)
            torch.set_num_threads(12)  # 使用所有逻辑核心
            
            with torch.no_grad(), torch.amp.autocast('cuda'):
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=input_data.get('do_sample', True),
                    streamer=streamer,
                    use_cache=True,
                    num_beams=1,
                    synced_gpus=False,
                    # 添加其他可用的优化参数
                    repetition_penalty=1.0,
                    length_penalty=1.0,
                    early_stopping=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_attention_mask=True
                )
            
            end_time = time.time()
            generation_time = end_time - start_time
            tokens_generated = len(outputs[0]) - len(input_ids[0])
            tokens_per_second = tokens_generated / generation_time
            
            logger.info(f"Generation completed in {generation_time:.2f} seconds")
            logger.info(f"Tokens generated: {tokens_generated}")
            logger.info(f"Generation speed: {tokens_per_second:.2f} tokens/second")
            logger.info(f"Final sequence length: {len(outputs[0])}")
            
            # 最终返回完整的 token ids
            return {
                'status': 'success',
                'output_ids': outputs.cpu().numpy().tolist()
            }
        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }

    def run(self):
        logger.info("Worker ready to receive requests")
        try:
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
                        logger.info("Received shutdown command")
                        self.socket.send_pyobj({'status': 'shutdown'})
                        # 清理资源
                        if hasattr(self, 'model'):
                            del self.model
                        if hasattr(self, 'tokenizer'):
                            del self.tokenizer
                        torch.cuda.empty_cache()
                        logger.info("Resources cleaned up")
                        break
                    
                except Exception as e:
                    logger.error(f"Error processing message: {str(e)}")
                    self.socket.send_pyobj({'status': 'error', 'error': str(e)})
        finally:
            # 确保资源被清理
            logger.info("Cleaning up worker resources...")
            self.socket.close()
            self.context.term()
            logger.info("Worker shutdown complete")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker_address', type=str, required=True)
    args = parser.parse_args()
    
    worker = ModelWorker(args.worker_address)
    worker.run()

if __name__ == '__main__':
    main() 