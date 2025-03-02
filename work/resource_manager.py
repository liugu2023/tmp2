from dataclasses import dataclass
from typing import List, Dict
import torch

@dataclass
class HardwareResource:
    host: str
    device_type: str  # 'cpu', 'gpu', 'memory'
    device_id: int = 0
    total_memory: int = 0  # MB
    model_path: str = ""
    gguf_path: str = ""
    
class ResourceManager:
    def __init__(self):
        self.resources: Dict[str, List[HardwareResource]] = {}
        
    def register_resource(self, resource: HardwareResource):
        if resource.host not in self.resources:
            self.resources[resource.host] = []
        self.resources[resource.host].append(resource)
    
    def get_available_resources(self):
        return self.resources
    
    def initialize_default_resources(self):
        # CPU 服务器配置
        self.register_resource(HardwareResource(
            host="10.21.22.100",
            device_type="cpu",
            total_memory=32768,  # 32GB内存
            model_path="/archive/liugu/ds/DeepSeek-R1-Distill-Llama-70B/",
            gguf_path="/archive/liugu/ds/DeepSeek-R1-Distill-Llama-70B-GGUF/DeepSeek-R1-Distill-Llama-70B-Q2_K.gguf"
        ))
        
        # GPU 服务器配置
        for gpu_id in range(2):
            self.register_resource(HardwareResource(
                host="10.21.22.206",
                device_type="gpu",
                device_id=gpu_id,
                model_path="/archive/liugu/ds/DeepSeek-R1-Distill-Llama-70B/",
                gguf_path="/archive/liugu/ds/DeepSeek-R1-Distill-Llama-70B-GGUF/DeepSeek-R1-Distill-Llama-70B-Q2_K.gguf"
            )) 