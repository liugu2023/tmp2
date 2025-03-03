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
import torch
from transformers import AutoTokenizer
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 只在非分布式模式下导入这些模块
def import_local_modules():
    from ktransformers.optimize.optimize import optimize_and_load_gguf
    from ktransformers.models.modeling_deepseek import DeepseekV2ForCausalLM
    from ktransformers.models.modeling_qwen2_moe import Qwen2MoeForCausalLM
    from ktransformers.models.modeling_deepseek_v3 import DeepseekV3ForCausalLM
    from ktransformers.models.modeling_llama import LlamaForCausalLM
    from ktransformers.models.modeling_mixtral import MixtralForCausalLM
    from ktransformers.util.utils import prefill_and_generate, get_compute_capability
    from ktransformers.server.config.config import Config
    from ktransformers.operators.flashinfer_wrapper import flashinfer_enabled
    return (optimize_and_load_gguf, DeepseekV2ForCausalLM, Qwen2MoeForCausalLM,
            DeepseekV3ForCausalLM, LlamaForCausalLM, MixtralForCausalLM,
            prefill_and_generate, get_compute_capability, Config, flashinfer_enabled)

class DistributedModel:
    def __init__(self, worker_address: str):
        self.worker_address = worker_address
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.connect(f"tcp://{worker_address}")
        self.socket = socket
        logger.info(f"Connected to worker at {worker_address}")
    
    def load_model(self, model_path: str, gguf_path: str):
        message = {
            'command': 'load_model',
            'model_path': model_path,
            'gguf_path': gguf_path
        }
        logger.info(f"Requesting model load from worker at {self.worker_address}...")
        self.socket.send_pyobj(message)
        response = self.socket.recv_pyobj()
        if response['status'] != 'success':
            raise Exception(f"Failed to load model on {self.worker_address}: {response.get('error')}")
        logger.info("Model loaded successfully on worker")

    def generate(self, input_ids, attention_mask, **kwargs):
        try:
            logger.info("Preparing generation request...")
            # 预先分配好 numpy 数组
            input_ids_np = input_ids.cpu().numpy()
            attention_mask_np = attention_mask.cpu().numpy()
            
            message = {
                'command': 'generate',
                'data': {
                    'input_ids': input_ids_np.tolist(),
                    'attention_mask': attention_mask_np.tolist(),
                    **kwargs
                }
            }
            
            # 发送请求
            logger.info("Sending request to worker...")
            send_start = time.time()
            self.socket.send_pyobj(message)
            
            # 接收响应
            logger.info("Waiting for response from worker...")
            response = self.socket.recv_pyobj()
            
            recv_time = time.time() - send_start
            logger.info(f"Network round trip took {recv_time:.2f} seconds")
            
            if response['status'] == 'success':
                # 合并所有 worker 的输出
                logger.info("Combining response from worker...")
                conv_start = time.time()
                
                output_tensor = torch.tensor(response['output_ids'])
                
                conv_time = time.time() - conv_start
                logger.info(f"Response combination took {conv_time:.2f} seconds")
                return output_tensor
            else:
                raise Exception(f"Generation failed on {self.worker_address}: {response.get('error')}")
            
        except Exception as e:
            logger.error(f"Error in generate: {str(e)}")
            raise

def main():
    # 设置终端编码
    if sys.platform.startswith('win'):
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stdin.reconfigure(encoding='utf-8')
    else:
        default_encoding = locale.getpreferredencoding()
        sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
        sys.stdin = open(sys.stdin.fileno(), mode='r', encoding=default_encoding, buffering=1)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--gguf_path', type=str, required=True)
    parser.add_argument('--worker_address', type=str, required=True,
                      help='Worker address (e.g., "10.21.22.206:29500")')
    parser.add_argument('--max_new_tokens', type=int, default=300)
    args = parser.parse_args()

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
                
            logger.info("Processing input...")
            if content == "":
                content = "Please write a piece of quicksort code in C++."
            elif os.path.isfile(content):
                logger.info(f"Reading from file: {content}")
                with open(content, "r", encoding='utf-8') as f:
                    content = f.read()
            
            logger.info("Creating chat messages...")
            messages = [{"role": "user", "content": content}]
            
            logger.info("Tokenizing input...")
            input_tensor = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            )
            
            logger.info("Starting generation...")
            outputs = model.generate(
                input_tensor,
                torch.ones_like(input_tensor),
                max_new_tokens=args.max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
            
            logger.info("Decoding response...")
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print("\nAssistant:", response)
        except UnicodeDecodeError as e:
            logger.error(f"Encoding error: {str(e)}")
            print("请使用UTF-8编码输入文本")
            continue
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            continue

if __name__ == "__main__":
    main()
