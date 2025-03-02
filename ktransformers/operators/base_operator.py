'''
Description  :  
Author       : Boxin Zhang
Version      : 0.1.0
Copyright (c) 2024 by KVCache.AI, All Rights Reserved. 
'''
from typing import Any
from torch import nn, Tensor
from ktransformers.util.custom_gguf import GGUFLoader
from transformers.configuration_utils import PretrainedConfig
import ktransformers.util.utils as utils
class BaseInjectedModule(nn.Module):
    
    def __init__(
        self,
        key: str,
        gguf_loader: GGUFLoader,
        config: PretrainedConfig,
        orig_module: nn.Module,
        device: str = "cuda",
        **kwargs
    ):
        super().__init__()
        # 首先设置 orig_module，因为其他属性可能依赖它
        object.__setattr__(self, "orig_module", orig_module)
        
        # 然后设置其他基本属性
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "gguf_loader", gguf_loader)
        object.__setattr__(self, "config", config)
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "prefill_device", device)
        object.__setattr__(self, "generate_device", device)

    def __getattr__(self, name: str) -> Any:
        try:
            return object.__getattribute__(self, name)
        except:
            if name == "orig_module":
                return nn.Module.__getattr__(self, "orig_module")
            try:
                return getattr(self.orig_module, name)
            except:
                return super(nn.Module, self.orig_module).__getattribute__(name)

    def __setattr__(self, name: str, value: Tensor | nn.Module) -> None:
        if name == "orig_module":
            object.__setattr__(self, name, value)
        elif hasattr(self, name):
            object.__setattr__(self, name, value)
        else:
            setattr(self.orig_module, name, value)
    
    def forward(self, *args, **kwargs):
        return self.orig_module.forward(*args, **kwargs)
    
    def load(self):
        for name, child in self._modules.items():
            utils.load_weights(child, self.gguf_loader, self.key+".")

