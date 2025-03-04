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
        try:
            logger.info("Preparing generation request...")
            message = {
                'command': 'generate',
                'data': {
                    'input_ids': input_ids.cpu().numpy().tolist(),
                    'attention_mask': attention_mask.cpu().numpy().tolist(),
                    **kwargs
                }
            }
            
            logger.info("Starting generation...")
            self.socket.send_pyobj(message)
            
            # 接收流式响应
            full_text = ""
            while True:
                response = self.socket.recv_pyobj()
                
                if response['status'] == 'error':
                    raise Exception(f"Generation failed: {response.get('error')}")
                
                if response['status'] == 'streaming':
                    # 打印新生成的文本部分
                    new_text = response['text'][len(full_text):]
                    if new_text:
                        print(new_text, end='', flush=True)
                    full_text = response['text']
                    
                    # 发送确认
                    self.socket.send_pyobj({'status': 'received'})
                    
                    # 如果生成结束，退出循环
                    if response.get('finished', False):
                        print()  # 换行
                        # 不要在这里 break，等待最终的 token IDs
                
                elif response['status'] == 'success':
                    # 最终的完整响应
                    return torch.tensor(response['output_ids'])
        
        except Exception as e:
            logger.error(f"Error in generate: {str(e)}")
            raise

    def shutdown(self):
        """发送关闭命令给worker"""
        try:
            logger.info("Sending shutdown command to worker...")
            message = {'command': 'shutdown'}
            self.socket.send_pyobj(message)
            response = self.socket.recv_pyobj()
            if response['status'] == 'shutdown':
                logger.info("Worker shutdown successfully")
            self.socket.close()
            self.context.term()
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

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
    parser.add_argument('--worker_address', type=str, default=None,
                      help='Address of worker for distributed mode (e.g., "10.21.22.206:29500")')
    parser.add_argument('--max_new_tokens', type=int, default=300)
    args = parser.parse_args()

    if args.worker_address:
        # 分布式模式
        logger.info("Running in distributed mode")
        model = DistributedModel(args.worker_address)
        model.load_model(args.model_path, args.gguf_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        
        try:
            while True:
                content = input("Chat: ")
                if content.lower() in ['quit', 'exit']:
                    logger.info("Shutting down...")
                    model.shutdown()  # 发送关闭命令
                    break
                elif content == "":
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
                
                # 不需要再次解码和打印，因为已经在流式输出中完成了
                if outputs is not None:  # 添加检查
                    logger.info("Generation completed successfully")
                else:
                    logger.warning("No output tokens received")
                
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, shutting down...")
            model.shutdown()
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            model.shutdown()

    else:
        # 本地模式，动态导入所需模块
        (optimize_and_load_gguf, DeepseekV2ForCausalLM, Qwen2MoeForCausalLM,
         DeepseekV3ForCausalLM, LlamaForCausalLM, MixtralForCausalLM,
         prefill_and_generate, get_compute_capability, Config, flashinfer_enabled) = import_local_modules()
        
        # 本地模式的代码...
        raise NotImplementedError("Local mode is not implemented in this version")

if __name__ == "__main__":
    main()
