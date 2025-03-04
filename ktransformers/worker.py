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
import psutil

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
        self.model_path = model_path  # 保存模型路径
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
        
        # 记录最终的设备分配计划
        self.layer_device_map = {
            'model.embed_tokens': 'cpu',
            'model.norm': 'cpu',
            'lm_head': 'cpu',
            **{f'model.layers.{i}': 'cpu' for i in range(cpu_front_layers)},
            **{f'model.layers.{i}': f'cuda:{(i-cpu_front_layers)//6}' 
               for i in range(cpu_front_layers, cpu_front_layers + gpu_middle_layers)},
            **{f'model.layers.{i}': 'cpu' 
               for i in range(cpu_front_layers + gpu_middle_layers, num_layers)}
        }
        
        logger.info("Loading model with dynamic loading...")
        # 首先加载空模型
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map='meta',  # 先全部放在 meta 设备
            low_cpu_mem_usage=True,
        ).to_empty(device='meta')
        
        # 创建模型参数存储目录
        os.makedirs("model_offload", exist_ok=True)
        
        # 初始加载必要的组件
        logger.info("Loading essential components...")
        self._load_essential_components()
        
        self.model.eval()
        logger.info("Model initialized successfully")
    
    def _load_essential_components(self):
        """加载必要的模型组件和常用词"""
        logger.info("Loading state dict...")
        try:
            # 从预训练模型加载实际权重
            state_dict = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                config=self.model.config,
                trust_remote_code=True,
                torch_dtype=torch.float32,  # 使用 float32 加载
                device_map=None,  # 不使用设备映射
                state_dict=True,  # 只加载状态字典
                low_cpu_mem_usage=True,
            ).state_dict()
            
            # 创建词表缓存目录
            vocab_cache_dir = "vocab_cache"
            os.makedirs(vocab_cache_dir, exist_ok=True)
            
            # 加载必要组件
            essential_components = ['model.embed_tokens', 'model.norm', 'lm_head']
            for name in essential_components:
                logger.info(f"Loading {name}")
                try:
                    module = self.model.get_submodule(name)
                    # 加载权重并移动到 CPU
                    for param_name, param in module.named_parameters():
                        full_name = f"{name}.{param_name}"
                        if full_name in state_dict:
                            try:
                                # 尝试加载到 CPU
                                param.data = state_dict[full_name].to('cpu')
                            except RuntimeError as e:
                                if "out of memory" in str(e):
                                    logger.warning(f"Not enough memory for {full_name}, saving to disk")
                                    # 保存到磁盘
                                    torch.save(state_dict[full_name], f"{vocab_cache_dir}/{full_name.replace('/', '_')}.pt")
                                    # 使用 meta tensor 占位
                                    param.data = torch.empty(1, device='meta')
                
                    self.loaded_layers.add(name)
                    logger.info(f"Loaded {name}")
                    
                except Exception as e:
                    logger.error(f"Error loading {name}: {str(e)}")
                    raise
            
            # 尝试加载常用词嵌入
            logger.info("Loading common token embeddings...")
            try:
                # 获取词嵌入矩阵
                embed_weight = state_dict['model.embed_tokens.weight']
                vocab_size = embed_weight.size(0)
                embed_dim = embed_weight.size(1)
                
                # 计算每个批次的大小（基于可用内存）
                available_mem = psutil.virtual_memory().available
                token_size = embed_dim * 4  # 每个token的近似内存大小（以字节为单位）
                batch_size = min(1000, max(100, available_mem // (token_size * 2)))  # 留出一半内存作为缓冲
                
                # 分批处理词嵌入
                for start_idx in range(0, vocab_size, batch_size):
                    end_idx = min(start_idx + batch_size, vocab_size)
                    batch_tokens = list(range(start_idx, end_idx))
                    
                    try:
                        # 尝试加载到内存
                        batch_embeds = embed_weight[batch_tokens].to('cpu')
                        # 保存到缓存文件
                        cache_file = f"{vocab_cache_dir}/token_embeds_{start_idx}_{end_idx}.pt"
                        torch.save(batch_embeds, cache_file)
                        logger.info(f"Cached tokens {start_idx}-{end_idx}")
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            logger.warning(f"Memory full at token {start_idx}, remaining tokens will be loaded on demand")
                            break
                        raise
                    
                    # 释放内存
                    del batch_embeds
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                logger.error(f"Error caching token embeddings: {str(e)}")
                logger.warning("Will load tokens on demand")
            
            # 清理原始状态字典以释放内存
            del state_dict
            torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"Error in essential components loading: {str(e)}")
            raise

    def _load_token_embedding(self, token_id: int):
        """按需加载单个词的嵌入"""
        vocab_cache_dir = "vocab_cache"
        
        # 确定token所在的缓存文件
        for cache_file in os.listdir(vocab_cache_dir):
            if cache_file.startswith("token_embeds_"):
                start_idx, end_idx = map(int, cache_file.split("_")[2:4])
                if start_idx <= token_id < end_idx:
                    try:
                        # 加载整个批次
                        batch_embeds = torch.load(f"{vocab_cache_dir}/{cache_file}")
                        # 获取特定token的嵌入
                        token_embed = batch_embeds[token_id - start_idx]
                        return token_embed
                    except Exception as e:
                        logger.error(f"Error loading token {token_id}: {str(e)}")
                        raise
        
        # 如果找不到缓存，从原始模型加载
        try:
            state_dict = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                config=self.model.config,
                trust_remote_code=True,
                state_dict=True,
                low_cpu_mem_usage=True,
            ).state_dict()
            
            token_embed = state_dict['model.embed_tokens.weight'][token_id]
            del state_dict
            return token_embed
            
        except Exception as e:
            logger.error(f"Error loading token {token_id} from model: {str(e)}")
            raise

    def _move_to_device(self, layer_name: str, device: str):
        """将指定层移动到目标设备"""
        try:
            # 从预训练模型加载实际权重
            state_dict = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                config=self.model.config,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map=None,
                state_dict=True,
                low_cpu_mem_usage=True,
            ).state_dict()
            
            # 获取模块
            module = self.model.get_submodule(layer_name)
            
            # 加载权重并移动到目标设备
            for param_name, param in module.named_parameters():
                full_name = f"{layer_name}.{param_name}"
                if full_name in state_dict:
                    param.data = state_dict[full_name].to(device)
            
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