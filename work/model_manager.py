import torch
from typing import Optional
import os

class DistributedModelManager:
    def __init__(self, model_path: str, gguf_path: str):
        self.model_path = model_path
        self.gguf_path = gguf_path
        self.model = None
        
    def load_model(self, device_type: str, device_id: Optional[int] = None):
        if device_type == "gpu":
            # GPU版本加载
            os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
            self.model = self._load_gpu_model()
        else:
            # CPU版本加载
            self.model = self._load_cpu_model()
            
    def _load_gpu_model(self):
        # 在GPU上加载原始模型
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map="auto",
            torch_dtype=torch.float16
        )
        return model
        
    def _load_cpu_model(self):
        # 在CPU上加载量化后的模型
        from ctransformers import AutoModelForCausalLM
        
        model = AutoModelForCausalLM.from_pretrained(
            self.gguf_path,
            model_type="llama"
        )
        return model 