import torch
from typing import Optional
from transformers import AutoTokenizer, AutoConfig, GenerationConfig, TextStreamer
from ktransformers.optimize.optimize import optimize_and_load_gguf
from ktransformers.util.utils import prefill_and_generate, get_compute_capability
from ktransformers.server.config.config import Config
from ktransformers.operators.flashinfer_wrapper import flashinfer_enabled

class DistributedWorker:
    def __init__(self, host: str, device_type: str, device_id: Optional[int] = None,
                 model_path: str = "", gguf_path: str = "", optimize_config_path: str = None):
        self.host = host
        self.device_type = device_type
        self.device_id = device_id
        self.device = self._setup_device()
        self.model_path = model_path
        self.gguf_path = gguf_path
        self.optimize_config_path = optimize_config_path
        self.model = None
        self.tokenizer = None
        self.config = None
        
    def initialize(self):
        # 设置设备
        if self.device_type == "gpu":
            torch.cuda.set_device(self.device_id)
        torch.set_grad_enabled(False)
        
        # 加载tokenizer和配置
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
        
        # 加载和优化模型
        from ktransformers.local_chat import custom_models, default_optimize_rules
        
        with torch.device("meta"):
            if self.config.architectures[0] in custom_models:
                if "Qwen2Moe" in self.config.architectures[0]:
                    self.config._attn_implementation = "flash_attention_2"
                if "Llama" in self.config.architectures[0]:
                    self.config._attn_implementation = "eager"
                if "Mixtral" in self.config.architectures[0]:
                    self.config._attn_implementation = "flash_attention_2"
                    
                self.model = custom_models[self.config.architectures[0]](self.config)
            else:
                self.model = AutoModelForCausalLM.from_config(
                    self.config, trust_remote_code=True, attn_implementation="flash_attention_2"
                )
        
        # 使用默认优化规则或指定的规则
        if not self.optimize_config_path and self.config.architectures[0] in default_optimize_rules:
            self.optimize_config_path = default_optimize_rules[self.config.architectures[0]]
            
        optimize_and_load_gguf(self.model, self.optimize_config_path, self.gguf_path, self.config)
        
        # 设置生成配置
        try:
            self.model.generation_config = GenerationConfig.from_pretrained(self.model_path)
        except Exception as e:
            print(f"使用默认生成配置。错误信息: {e}")
            self.model.generation_config = GenerationConfig(
                temperature=0.6,
                top_p=0.95,
                do_sample=True
            )
        
        if self.model.generation_config.pad_token_id is None:
            self.model.generation_config.pad_token_id = self.model.generation_config.eos_token_id
            
        self.model.eval()
        
    def generate_response(self, prompt: str, max_new_tokens: int = 300, 
                         mode: str = "normal", force_think: bool = False):
        messages = [{"role": "user", "content": prompt}]
        input_tensor = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
        
        if force_think:
            token_thinks = torch.tensor(
                [self.tokenizer.encode("<think>\\n", add_special_tokens=False)],
                device=input_tensor.device
            )
            input_tensor = torch.cat([input_tensor, token_thinks], dim=1)
            
        # 使用flashinfer优化（如果可用）
        if (platform.system() != "Windows" and 
            self.config.architectures[0] in ["DeepseekV2ForCausalLM", "DeepseekV3ForCausalLM"] and 
            flashinfer_enabled and 
            get_compute_capability() >= 8):
            
            generated = prefill_and_generate(
                self.model, 
                self.tokenizer,
                input_tensor.cuda(),
                max_new_tokens,
                use_cuda_graph=True,
                mode=mode,
                force_think=force_think,
                use_flashinfer_mla=True,
                num_heads=self.config.num_attention_heads,
                head_dim_ckv=self.config.kv_lora_rank,
                head_dim_kpe=self.config.qk_rope_head_dim,
                q_head_dim=self.config.qk_rope_head_dim + self.config.qk_nope_head_dim
            )
        else:
            generated = prefill_and_generate(
                self.model,
                self.tokenizer,
                input_tensor.cuda(),
                max_new_tokens,
                use_cuda_graph=True,
                mode=mode,
                force_think=force_think,
            )
        
        return generated
        
    def _setup_device(self):
        if self.device_type == "gpu":
            return torch.device(f"cuda:{self.device_id}")
        return torch.device("cpu")
    
    def execute_task(self, task_fn, *args, **kwargs):
        """
        在指定设备上执行任务
        """
        with torch.device(self.device):
            return task_fn(*args, **kwargs) 