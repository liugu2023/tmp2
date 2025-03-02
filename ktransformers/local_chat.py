"""
Description  :  
Author       : Boxin Zhang, Azure-Tang
Version      : 0.1.0
Copyright (c) 2024 by KVCache.AI, All Rights Reserved. 
"""

import os
import platform
import sys
import zmq
import argparse
import logging
import locale

project_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_dir)
import torch
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    GenerationConfig,
    TextStreamer,
)
import json
import fire
from ktransformers.optimize.optimize import optimize_and_load_gguf
from ktransformers.models.modeling_deepseek import DeepseekV2ForCausalLM
from ktransformers.models.modeling_qwen2_moe import Qwen2MoeForCausalLM
from ktransformers.models.modeling_deepseek_v3 import DeepseekV3ForCausalLM
from ktransformers.models.modeling_llama import LlamaForCausalLM
from ktransformers.models.modeling_mixtral import MixtralForCausalLM
from ktransformers.util.utils import prefill_and_generate, get_compute_capability
from ktransformers.server.config.config import Config
from ktransformers.operators.flashinfer_wrapper import flashinfer_enabled

custom_models = {
    "DeepseekV2ForCausalLM": DeepseekV2ForCausalLM,
    "DeepseekV3ForCausalLM": DeepseekV3ForCausalLM,
    "Qwen2MoeForCausalLM": Qwen2MoeForCausalLM,
    "LlamaForCausalLM": LlamaForCausalLM,
    "MixtralForCausalLM": MixtralForCausalLM,
}

ktransformer_rules_dir = (
    os.path.dirname(os.path.abspath(__file__)) + "/optimize/optimize_rules/"
)
default_optimize_rules = {
    "DeepseekV2ForCausalLM": ktransformer_rules_dir + "DeepSeek-V2-Chat.yaml",
    "DeepseekV3ForCausalLM": ktransformer_rules_dir + "DeepSeek-V3-Chat.yaml",
    "Qwen2MoeForCausalLM": ktransformer_rules_dir + "Qwen2-57B-A14B-Instruct.yaml",
    "LlamaForCausalLM": ktransformer_rules_dir + "Internlm2_5-7b-Chat-1m.yaml",
    "MixtralForCausalLM": ktransformer_rules_dir + "Mixtral.yaml",
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DistributedModel:
    def __init__(self, worker_address: str):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{worker_address}")
        logger.info(f"Connected to worker at {worker_address}")
        
    def load_model(self, model_path: str, gguf_path: str):
        message = {
            'command': 'load_model',
            'model_path': model_path,
            'gguf_path': gguf_path
        }
        logger.info("Requesting model load from worker...")
        self.socket.send_pyobj(message)
        response = self.socket.recv_pyobj()
        if response['status'] != 'success':
            raise Exception(f"Failed to load model: {response.get('error')}")
        logger.info("Model loaded successfully on worker")

    def generate(self, input_ids, attention_mask, **kwargs):
        message = {
            'command': 'generate',
            'data': {
                'input_ids': input_ids.cpu().numpy().tolist(),
                'attention_mask': attention_mask.cpu().numpy().tolist(),
                **kwargs
            }
        }
        self.socket.send_pyobj(message)
        response = self.socket.recv_pyobj()
        
        if response['status'] == 'success':
            return torch.tensor(response['output_ids'])
        else:
            raise Exception(f"Generation failed: {response.get('error')}")

def local_chat(
    model_path: str | None = None,
    optimize_config_path: str = None,
    gguf_path: str | None = None,
    max_new_tokens: int = 300,
    cpu_infer: int = Config().cpu_infer,
    use_cuda_graph: bool = True,
    prompt_file : str | None = None,
    mode: str = "normal",
    force_think: bool = False,
    chunk_prefill_size: int = 8192
):

    torch.set_grad_enabled(False)

    Config().cpu_infer = cpu_infer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if mode == 'long_context':
        assert config.architectures[0] == "LlamaForCausalLM", "only LlamaForCausalLM support long_context mode"
        torch.set_default_dtype(torch.float16)
    else:
        torch.set_default_dtype(config.torch_dtype)

    with torch.device("meta"):
        if config.architectures[0] in custom_models:
            print("using custom modeling_xxx.py.")
            if (
                "Qwen2Moe" in config.architectures[0]
            ):  # Qwen2Moe must use flash_attention_2 to avoid overflow.
                config._attn_implementation = "flash_attention_2"
            if "Llama" in config.architectures[0]:
                config._attn_implementation = "eager"
            if "Mixtral" in config.architectures[0]:
                config._attn_implementation = "flash_attention_2"

            model = custom_models[config.architectures[0]](config)
        else:
            model = AutoModelForCausalLM.from_config(
                config, trust_remote_code=True, attn_implementation="flash_attention_2"
            )

    if optimize_config_path is None:
        if config.architectures[0] in default_optimize_rules:
            print("using default_optimize_rule for", config.architectures[0])
            optimize_config_path = default_optimize_rules[config.architectures[0]]
        else:
            optimize_config_path = input(
                "please input the path of your rule file(yaml file containing optimize rules):"
            )

    if gguf_path is None:
        gguf_path = input(
            "please input the path of your gguf file(gguf file in the dir containing input gguf file must all belong to current model):"
        )
    optimize_and_load_gguf(model, optimize_config_path, gguf_path, config)
    
    try:
        model.generation_config = GenerationConfig.from_pretrained(model_path)
    except Exception as e:
        print(f"generation config can't auto create, make default. Message: {e}")
        gen_config = GenerationConfig(
            temperature=0.6,
            top_p=0.95,
            do_sample=True
        )
        model.generation_config = gen_config
    # model.generation_config = GenerationConfig.from_pretrained(model_path)
    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = model.generation_config.eos_token_id
    model.eval()

    system = platform.system()
    if system == "Windows":
        os.system("cls")
    else:
        os.system("clear")

    while True:
        content = input("Chat: ")
        if content.startswith('"""'):  # prefix """
            # multi lines input
            content = content[3:] + "\n"
            while True:
                line = input("")
                if line.endswith('"""'):
                    # end multi lines input
                    line = line[:-3]  # suffix """
                    if line:
                        content += line + "\n"
                    break
                else:
                    content += line + "\n"

        if content == "":
            if prompt_file != None:
                content = open(prompt_file, "r").read()
            else:
                content = "Please write a piece of quicksort code in C++."
        elif os.path.isfile(content):
            content = open(content, "r").read()
            
        messages = [{"role": "user", "content": content}]
        input_tensor = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
        if force_think:
            token_thinks = torch.tensor([tokenizer.encode("<think>\\n",add_special_tokens=False)],device=input_tensor.device)
            input_tensor = torch.cat(
                [input_tensor, token_thinks], dim=1
            )
        if mode == 'long_context':
            assert Config().long_context_config['max_seq_len'] > input_tensor.shape[1] + max_new_tokens, \
            "please change max_seq_len in  ~/.ktransformers/config.yaml"
        
        if system != "Windows" and (config.architectures[0] == "DeepseekV2ForCausalLM" or config.architectures[0] == "DeepseekV3ForCausalLM") and flashinfer_enabled and get_compute_capability() >= 8:
            generated = prefill_and_generate(
                model, tokenizer, input_tensor.cuda(), max_new_tokens, use_cuda_graph, mode = mode, force_think = force_think, chunk_prefill_size = chunk_prefill_size,
                use_flashinfer_mla = True, num_heads = config.num_attention_heads, head_dim_ckv = config.kv_lora_rank, head_dim_kpe = config.qk_rope_head_dim, q_head_dim = config.qk_rope_head_dim + config.qk_nope_head_dim
            )
        else:
            generated = prefill_and_generate(
                model, tokenizer, input_tensor.cuda(), max_new_tokens, use_cuda_graph, mode = mode, force_think = force_think, chunk_prefill_size = chunk_prefill_size,
            )

def main():
    # 设置终端编码
    if sys.platform.startswith('win'):
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stdin.reconfigure(encoding='utf-8')
    else:
        # 获取系统默认编码
        default_encoding = locale.getpreferredencoding()
        sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
        sys.stdin = open(sys.stdin.fileno(), mode='r', encoding=default_encoding, buffering=1)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--gguf_path', type=str, required=True)
    parser.add_argument('--worker_address', type=str, default=None,
                      help='Address of worker for distributed mode (e.g., "10.21.22.206:29500")')
    parser.add_argument('--max_new_tokens', type=int, default=300)
    parser.add_argument('--cpu_infer', type=int, default=Config().cpu_infer)
    parser.add_argument('--use_cuda_graph', type=bool, default=True)
    parser.add_argument('--prompt_file', type=str, default=None)
    parser.add_argument('--mode', type=str, default="normal")
    parser.add_argument('--force_think', type=bool, default=False)
    parser.add_argument('--chunk_prefill_size', type=int, default=8192)
    args = parser.parse_args()

    if args.worker_address:
        # 分布式模式
        logger.info("Running in distributed mode")
        model = DistributedModel(args.worker_address)
        model.load_model(args.model_path, args.gguf_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        
        while True:
            try:
                content = input("Chat: ")
                if content.lower() in ['quit', 'exit']:
                    break
                    
                if content == "":
                    if args.prompt_file is not None:
                        with open(args.prompt_file, "r", encoding='utf-8') as f:
                            content = f.read()
                    else:
                        content = "Please write a piece of quicksort code in C++."
                elif os.path.isfile(content):
                    with open(content, "r", encoding='utf-8') as f:
                        content = f.read()
                
                messages = [{"role": "user", "content": content}]
                input_tensor = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, return_tensors="pt"
                )
                
                outputs = model.generate(
                    input_tensor,
                    torch.ones_like(input_tensor),
                    max_new_tokens=args.max_new_tokens,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True
                )
                
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                print("Assistant:", response)
            except UnicodeDecodeError as e:
                logger.error(f"Encoding error: {str(e)}")
                print("请使用UTF-8编码输入文本")
                continue
            except Exception as e:
                logger.error(f"Error: {str(e)}")
                continue

    else:
        # 本地模式
        local_chat(
            model_path=args.model_path,
            gguf_path=args.gguf_path,
            max_new_tokens=args.max_new_tokens,
            cpu_infer=args.cpu_infer,
            use_cuda_graph=args.use_cuda_graph,
            prompt_file=args.prompt_file,
            mode=args.mode,
            force_think=args.force_think,
            chunk_prefill_size=args.chunk_prefill_size
        )

if __name__ == "__main__":
    fire.Fire(main)
