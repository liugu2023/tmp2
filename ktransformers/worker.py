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
        num_layers = config.num_hidden_layers
        logger.info(f"Total layers: {num_layers}")
        
        # 调整层分配
        cpu_front_layers = 34  # 前34层放在CPU
        gpu_middle_layers = 12  # GPU总共处理12层(每个GPU 6层)
        cpu_end_layers = num_layers - cpu_front_layers - gpu_middle_layers
        
        # 创建详细的设备映射
        device_map = {
            'model.embed_tokens': 'cpu',
            'model.norm': 'cpu',
            'lm_head': 'cpu',
        }
        
        # 分配层到不同设备
        for i in range(num_layers):
            if i < cpu_front_layers:
                device_map[f'model.layers.{i}'] = 'cpu'
            elif i < cpu_front_layers + gpu_middle_layers:
                gpu_id = (i - cpu_front_layers) // (gpu_middle_layers // 2)
                device_map[f'model.layers.{i}'] = f'cuda:{gpu_id}'
            else:
                device_map[f'model.layers.{i}'] = 'cpu'
        
        # 设置内存限制
        max_memory = {
            0: "12GiB",
            1: "12GiB",
            'cpu': "138GiB"
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
        
        # 加载模型时直接设置 dtype
        model_kwargs = {
            'device_map': device_map,
            'max_memory': max_memory,
            'low_cpu_mem_usage': True,
            'torch_dtype': {
                'cpu': torch.float32,
                'cuda:0': torch.float16,
                'cuda:1': torch.float16
            }
        }
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            trust_remote_code=True,
            **model_kwargs
        )
        
        self.model.eval()
        logger.info("Model loaded successfully with custom device mapping")