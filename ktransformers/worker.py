import torch
import zmq
import pickle
from typing import Dict, Any, List
import logging
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, GenerationConfig, TextStreamer
import time
import os
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

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

class KVCacheServer:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.kv_cache = {}
        self.lock = threading.Lock()
        
    def load_model(self, model_path: str, gguf_path: str):
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        logger.info("Loading model config...")
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        config.attn_implementation = "flash_attention_2"
        torch.set_default_dtype(config.torch_dtype)
        
        logger.info("Loading model...")
        
        # 计算模型层数
        num_layers = config.num_hidden_layers  # 应该是 80 层
        logger.info(f"Total layers: {num_layers}")
        
        # 调整层分配
        cpu_front_layers = 34  # 前34层放在CPU
        gpu_middle_layers = 12  # GPU总共处理12层(每个GPU 6层)
        cpu_end_layers = num_layers - cpu_front_layers - gpu_middle_layers  # 剩余层放在CPU
        
        # 创建详细的设备映射
        device_map = {
            'model.embed_tokens': 'cpu',  # embedding 层放在 CPU
            'model.norm': 'cpu',  # 最终的 norm 层放在 CPU
            'lm_head': 'cpu',  # 输出层放在 CPU
        }
        
        # 前34层放在 CPU
        for i in range(cpu_front_layers):
            device_map[f'model.layers.{i}'] = 'cpu'
        
        # 中间12层在两个 GPU 之间平衡分配（每个GPU 6层）
        gpu_layers_per_device = gpu_middle_layers // 2  # 每个GPU 6层
        for i in range(cpu_front_layers, cpu_front_layers + gpu_middle_layers):
            gpu_id = (i - cpu_front_layers) // gpu_layers_per_device
            device_map[f'model.layers.{i}'] = f'cuda:{gpu_id}'
        
        # 剩余层放在 CPU
        for i in range(cpu_front_layers + gpu_middle_layers, num_layers):
            device_map[f'model.layers.{i}'] = 'cpu'
        
        # 设置内存限制
        max_memory = {
            0: "12GiB",  # 第一个GPU分配12GB
            1: "12GiB",  # 第二个GPU分配12GB
            'cpu': "138GiB"  # CPU内存
        }
        
        logger.info("Device mapping summary:")
        logger.info(f"CPU front layers: 0-{cpu_front_layers-1}")
        logger.info(f"GPU layers: {cpu_front_layers}-{cpu_front_layers+gpu_middle_layers-1}")
        logger.info(f"CPU end layers: {cpu_front_layers+gpu_middle_layers}-{num_layers-1}")
        
        # 详细的设备映射日志
        device_counts = {'cpu': 0, 'cuda:0': 0, 'cuda:1': 0}
        for layer, device in device_map.items():
            logger.info(f"{layer}: {device}")
            device_counts[device] += 1
        
        logger.info("\nLayer distribution:")
        for device, count in device_counts.items():
            logger.info(f"{device}: {count} layers")
        
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map=device_map,
            max_memory=max_memory,
            low_cpu_mem_usage=True,
        )
        
        logger.info("Converting CPU layers to float32...")
        # 获取需要转换的模块列表
        cpu_modules = []
        for name, module in self.model.named_modules():
            if any(device == 'cpu' for layer, device in device_map.items() if layer in name):
                cpu_modules.append((name, module))
        
        # 分批处理转换
        batch_size = 4  # 每次处理4个模块
        for i in range(0, len(cpu_modules), batch_size):
            batch = cpu_modules[i:i + batch_size]
            for name, module in batch:
                logger.info(f"Converting {name} to float32")
                try:
                    # 对于大型模块，进一步分批处理参数
                    for param_name, param in module.named_parameters():
                        if param.device.type == 'cuda':
                            # 如果参数在 GPU 上，先移到 CPU
                            param.data = param.data.cpu()
                        param.data = param.data.to(torch.float32)
                    
                    # 确保模块被正确分配到指定设备
                    target_device = next(
                        device for layer, device in device_map.items() 
                        if layer in name
                    )
                    if 'cuda' in target_device:
                        gpu_id = int(target_device.split(':')[1])
                        # 在目标 GPU 上分配内存
                        with torch.cuda.device(gpu_id):
                            module.to(target_device)
                    else:
                        module.to('cpu')
                    
                except Exception as e:
                    logger.error(f"Error converting {name}: {str(e)}")
                    raise
            
            # 清理未使用的内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        logger.info("Model conversion completed")
        self.model.eval()
        
        # 打印最终的内存使用情况
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                memory_stats = torch.cuda.memory_stats(i)
                allocated = memory_stats['allocated_bytes.all.current'] / 1024**3
                reserved = memory_stats['reserved_bytes.all.current'] / 1024**3
                logger.info(f"GPU {i} memory usage:")
                logger.info(f"  Allocated: {allocated:.2f} GB")
                logger.info(f"  Reserved: {reserved:.2f} GB")
        
        logger.info("Model loaded successfully with custom device mapping")

class ModelWorker:
    def __init__(self, address: str):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://{address}")
        self.kv_server = KVCacheServer()
        logger.info(f"Worker started at {address}")

    def generate(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            with self.kv_server.lock:
                with torch.no_grad():
                    outputs = self.kv_server.model.generate(
                        input_ids=input_data['input_ids'],
                        attention_mask=input_data['attention_mask'],
                        max_new_tokens=input_data.get('max_new_tokens', 512),
                        temperature=input_data.get('temperature', 0.6),
                        top_p=input_data.get('top_p', 0.9),
                        do_sample=input_data.get('do_sample', True),
                        use_cache=True,
                        num_beams=1,
                        repetition_penalty=1.0,
                        length_penalty=1.0,
                        pad_token_id=self.kv_server.tokenizer.pad_token_id,
                        eos_token_id=self.kv_server.tokenizer.eos_token_id
                    )
            
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
        logger.info("Worker ready")
        try:
            while True:
                message = self.socket.recv_pyobj()
                command = message.get('command')
                
                if command == 'load_model':
                    self.kv_server.load_model(
                        message['model_path'],
                        message['gguf_path']
                    )
                    self.socket.send_pyobj({'status': 'success'})
                
                elif command == 'generate':
                    result = self.generate(message['data'])
                    self.socket.send_pyobj(result)
                
                elif command == 'shutdown':
                    self.socket.send_pyobj({'status': 'shutdown'})
                    break
                
        finally:
            if hasattr(self, 'kv_server'):
                del self.kv_server
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