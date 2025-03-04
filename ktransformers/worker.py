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
        self.param_usage_stats = {}  # 记录参数使用频率
        self.vocab_usage_stats = {}  # 记录词汇使用频率
        self.loaded_layers = set()  # 记录已加载的层
        
    def load_model(self, model_path: str, gguf_path: str):
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        logger.info("Loading model config...")
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        config.attn_implementation = "flash_attention_2"
        torch.set_default_dtype(config.torch_dtype)
        
        # 计算模型层数
        num_layers = config.num_hidden_layers
        logger.info(f"Total layers: {num_layers}")
        
        # 调整层分配
        cpu_front_layers = 34  # 前34层放在CPU
        gpu_middle_layers = 12  # GPU总共处理12层(每个GPU 6层)
        cpu_end_layers = num_layers - cpu_front_layers - gpu_middle_layers
        
        # 创建详细的设备映射
        device_map = {
            'model.embed_tokens': 'meta',  # 先放在 meta 设备上
            'model.norm': 'meta',
            'lm_head': 'meta',
        }
        
        # 所有层初始都放在 meta 设备上
        for i in range(num_layers):
            device_map[f'model.layers.{i}'] = 'meta'
        
        # 设置内存限制
        max_memory = {
            0: "12GiB",
            1: "12GiB",
            'cpu': "138GiB"
        }
        
        # 记录最终的设备分配计划
        self.layer_device_map = {
            **{f'model.layers.{i}': 'cpu' for i in range(cpu_front_layers)},
            **{f'model.layers.{i}': f'cuda:{(i-cpu_front_layers)//6}' 
               for i in range(cpu_front_layers, cpu_front_layers + gpu_middle_layers)},
            **{f'model.layers.{i}': 'cpu' 
               for i in range(cpu_front_layers + gpu_middle_layers, num_layers)}
        }
        
        logger.info("Loading model with dynamic loading...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map=device_map,  # 初始全部放在 meta 设备
            max_memory=max_memory,
            low_cpu_mem_usage=True,
        )
        
        # 创建模型参数存储目录
        os.makedirs("model_offload", exist_ok=True)
        
        # 初始加载必要的组件
        logger.info("Loading essential components...")
        self._load_essential_components()
        
        self.model.eval()
        logger.info("Model initialized successfully")
    
    def _load_essential_components(self):
        """加载必要的模型组件"""
        # 加载 embedding 层
        self._move_to_device('model.embed_tokens', 'cpu')
        # 加载最终的 norm 层
        self._move_to_device('model.norm', 'cpu')
        # 加载输出层
        self._move_to_device('lm_head', 'cpu')
        
        # 转换为 float32 以提高稳定性
        self.model.embed_tokens.to(torch.float32)
        self.model.norm.to(torch.float32)
        self.model.lm_head.to(torch.float32)
    
    def _move_to_device(self, layer_name: str, device: str):
        """将指定层移动到目标设备"""
        try:
            module = self.model.get_submodule(layer_name)
            module.to(device)
            self.loaded_layers.add(layer_name)
            logger.info(f"Moved {layer_name} to {device}")
        except Exception as e:
            logger.error(f"Error moving {layer_name} to {device}: {str(e)}")
            raise
    
    def _ensure_layer_loaded(self, layer_name: str):
        """确保指定层已加载到正确的设备"""
        if layer_name not in self.loaded_layers:
            target_device = self.layer_device_map.get(layer_name)
            if target_device:
                self._move_to_device(layer_name, target_device)
                self._maybe_offload_params()
    
    def _maybe_offload_params(self):
        """如果需要，卸载最少使用的参数"""
        if torch.cuda.is_available():
            # 检查 GPU 内存使用情况
            for i in range(torch.cuda.device_count()):
                memory_stats = torch.cuda.memory_stats(i)
                allocated = memory_stats['allocated_bytes.all.current'] / 1024**3
                
                # 如果使用超过 80%，开始卸载
                if allocated > 9.6:  # 12GB * 0.8
                    self._offload_least_used_params(i)
    
    def _offload_least_used_params(self, gpu_id):
        """卸载指定 GPU 上最少使用的参数"""
        # 获取该 GPU 上的参数
        gpu_params = [
            p for p in self.loaded_layers
            if self._get_param_device(p) == f'cuda:{gpu_id}'
        ]
        
        # 按最后访问时间排序
        gpu_params.sort(key=lambda p: self.param_usage_stats.get(p, 0))
        
        # 卸载 20% 最少使用的参数
        num_to_offload = max(1, len(gpu_params) // 5)
        for param_name in gpu_params[:num_to_offload]:
            self._offload_param(param_name)
    
    def _offload_param(self, param_name):
        """将参数卸载到 CPU 或磁盘"""
        logger.info(f"Offloading parameter {param_name}")
        param_data = self.model.state_dict()[param_name]
        
        # 保存到磁盘
        torch.save(param_data, f"model_offload/{param_name}.bin")
        
        # 从 GPU 移除
        self.model.state_dict()[param_name].data = torch.empty(1, device='meta')
        self.loaded_layers.remove(param_name)
        torch.cuda.empty_cache()

    def generate(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        # 在生成前确保需要的参数已加载
        self._ensure_required_params(input_data)
        
        try:
            with self.kv_cache.lock:
                with torch.no_grad():
                    outputs = self.model.generate(
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
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
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

class ModelWorker:
    def __init__(self, address: str):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://{address}")
        self.kv_server = KVCacheServer()
        logger.info(f"Worker started at {address}")

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
                    result = self.kv_server.generate(message['data'])
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