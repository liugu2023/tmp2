import torch
import zmq
import pickle
from typing import Dict, Any
import logging
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, GenerationConfig
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
        self.device_map = None  # 添加设备映射跟踪
        self.n_layers = None    # 添加层数跟踪
        logger.info(f"Worker started at {address}, listening on all interfaces")

    def load_model(self, model_path: str, gguf_path: str):
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        logger.info("Loading model config...")
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        torch.set_default_dtype(config.torch_dtype)
        self.n_layers = config.num_hidden_layers
        
        logger.info("Loading model with default device mapping...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16
        )
        self.model.eval()
        
        # 设置生成配置
        try:
            self.model.generation_config = GenerationConfig.from_pretrained(model_path)
        except Exception as e:
            logger.warning(f"Generation config can't auto create, making default. Message: {e}")
            self.model.generation_config = GenerationConfig(
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
        if self.model.generation_config.pad_token_id is None:
            self.model.generation_config.pad_token_id = self.model.generation_config.eos_token_id
        
        # 打印每个设备的内存使用情况
        logger.info("Initial memory usage after model loading:")
        logger.info("GPU Memory Usage:")
        for i in range(torch.cuda.device_count()):
            mem_allocated = torch.cuda.memory_allocated(i) / 1024**2
            mem_reserved = torch.cuda.memory_reserved(i) / 1024**2
            logger.info(f"  GPU {i}: Allocated: {mem_allocated:.1f}MB, Reserved: {mem_reserved:.1f}MB")
        
        logger.info("Model loaded successfully")

    def _optimize_memory(self, input_length: int):
        """根据输入长度动态调整模型层的设备分配"""
        if input_length > 512:  # 对于长输入，将更多层移到 CPU
            n_layers_gpu = int(self.n_layers * 0.5)  # 50% 的层在 GPU
        else:
            n_layers_gpu = int(self.n_layers * 0.7)  # 70% 的层在 GPU
        
        new_device_map = {
            'model.embed_tokens': 'cuda',
            'model.norm': 'cuda',
            'lm_head': 'cuda',
        }
        
        for i in range(self.n_layers):
            if i < n_layers_gpu:
                new_device_map[f'model.layers.{i}'] = 'cuda'
            else:
                new_device_map[f'model.layers.{i}'] = 'cpu'
        
        # 只在设备映射发生变化时才重新分配
        if new_device_map != self.device_map:
            logger.info("Adjusting model layer distribution...")
            logger.info(f"Moving {self.n_layers - n_layers_gpu} layers to CPU")
            
            # 记录调整前的内存
            before_mem = torch.cuda.memory_allocated() / 1024**2
            
            # 移动层到新设备
            for name, device in new_device_map.items():
                if hasattr(self.model, name):
                    module = getattr(self.model, name)
                    module.to(device)
                else:
                    try:
                        module = self.model.get_submodule(name)
                        module.to(device)
                    except AttributeError:
                        continue
            
            # 记录调整后的内存
            after_mem = torch.cuda.memory_allocated() / 1024**2
            logger.info(f"Memory change: {after_mem - before_mem:.1f}MB")
            
            self.device_map = new_device_map
            torch.cuda.empty_cache()

    def generate(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            logger.info("Received generation request")
            input_length = len(input_data['input_ids'])
            logger.info(f"Input sequence length: {input_length}")
            
            # 根据输入长度优化内存
            self._optimize_memory(input_length)
            
            # 创建输入张量
            logger.info("Moving input tensors to GPU...")
            input_ids = torch.tensor(input_data['input_ids'], device='cuda')
            attention_mask = torch.tensor(input_data['attention_mask'], device='cuda')
            
            # 生成参数日志
            temperature = input_data.get('temperature', 0.6)
            logger.info(f"Generation parameters: max_new_tokens={input_data.get('max_new_tokens', 512)}, "
                       f"temperature={temperature}, "
                       f"top_p={input_data.get('top_p', 0.9)}")
            
            # 生成文本
            logger.info("Starting text generation...")
            logger.info("Model configuration:")
            logger.info(f"  Model dtype: {self.model.dtype}")
            logger.info(f"  Device: {self.model.device}")
            logger.info(f"  Generation config: {self.model.generation_config}")
            
            def log_callback(step: int, token_ids: torch.Tensor, scores: torch.Tensor):
                logger.info(f"Generation step {step}: Generated {len(token_ids[0])} tokens")
                if scores is not None:
                    logger.info(f"  Top token score: {scores[0].max().item():.3f}")
            
            with torch.no_grad(), torch.amp.autocast('cuda'):
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=input_data.get('max_new_tokens', 512),
                    temperature=temperature,
                    top_p=input_data.get('top_p', 0.9),
                    do_sample=input_data.get('do_sample', True),
                    callback=log_callback,
                    callback_interval=50
                )
            
            logger.info(f"Generation completed. Output sequence length: {len(outputs[0])}")
            logger.info(f"New tokens generated: {len(outputs[0]) - len(input_ids[0])}")
            
            # 移动到CPU
            logger.info("Moving outputs to CPU...")
            outputs_cpu = outputs.to('cpu', non_blocking=True)
            torch.cuda.synchronize()
            
            # 记录内存使用情况
            logger.info("GPU Memory Usage:")
            for i in range(torch.cuda.device_count()):
                mem_allocated = torch.cuda.memory_allocated(i) / 1024**2
                mem_reserved = torch.cuda.memory_reserved(i) / 1024**2
                logger.info(f"  GPU {i}: Allocated: {mem_allocated:.1f}MB, Reserved: {mem_reserved:.1f}MB")
            
            logger.info("Preparing response...")
            return {
                'output_ids': outputs_cpu.numpy().tolist(),
                'status': 'success'
            }
        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            logger.exception("Detailed traceback:")
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
    parser.add_argument('--num_threads', type=int, default=10)
    args = parser.parse_args()
    
    # 设置 CPU 线程数
    torch.set_num_threads(args.num_threads)
    logger.info(f"Set CPU threads to {args.num_threads}")
    
    worker = ModelWorker(args.worker_address)
    worker.run()

if __name__ == '__main__':
    main() 