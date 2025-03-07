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
    def __init__(self, worker_address: str = None, scheduler_address: str = None):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        
        if scheduler_address:
            # 连接到调度器
            self.socket.connect(f"tcp://{scheduler_address}")
            logger.info(f"Connected to scheduler at {scheduler_address}")
            self.use_scheduler = True
        elif worker_address:
            # 直接连接到worker
            self.socket.connect(f"tcp://{worker_address}")
            logger.info(f"Connected to worker at {worker_address}")
            self.use_scheduler = False
        else:
            raise ValueError("Must provide either worker_address or scheduler_address")

        # 初始化后检查模型状态
        if self.use_scheduler:
            self._check_model_status()

    def _check_model_status(self):
        """检查模型是否已加载，避免重复加载"""
        try:
            logger.info("Checking model status...")
            self.socket.send_pyobj({"command": "status"})
            response = self.socket.recv_pyobj()
            
            if response.get('status') == 'success' and response.get('model_loaded', False):
                logger.info("Model already loaded, skipping load step")
                self.model_loaded = True
                print("模型已经加载完成，可以开始对话")
                return True
            else:
                logger.info("Model not loaded yet")
                return False
        except Exception as e:
            logger.error(f"Error checking model status: {e}")
            return False

    def load_model(self, model_path: str, gguf_path: str):
        if self.model_loaded:
            logger.info("Model already loaded, skipping load step")
            return True
        
        # 先检查模型状态，避免重复加载
        if self.use_scheduler and self._check_model_status():
            return True
        
        message = {
            'command': 'load_model',
            'model_path': model_path,
            'gguf_path': gguf_path
        }
        logger.info("Requesting model load...")
        self.socket.send_pyobj(message)
        response = self.socket.recv_pyobj()
        if response['status'] != 'success':
            raise Exception(f"Failed to load model: {response.get('error')}")
        logger.info("Model loaded successfully")

    def generate_from_messages(self, messages, **kwargs):
        """使用聊天消息直接生成文本，由调度器处理tokenization"""
        try:
            if not self.use_scheduler:
                raise ValueError("This method requires a scheduler connection")
                
            logger.info("Preparing chat request...")
            message = {
                'command': 'chat',
                'messages': messages,
                **kwargs
            }
            
            logger.info("Sending chat request to scheduler...")
            self.socket.send_pyobj(message)
            
            # 接收流式响应
            full_text = ""
            while True:
                response = self.socket.recv_pyobj()
                
                if response['status'] == 'error':
                    raise Exception(f"Generation failed: {response.get('error')}")
                
                elif response['status'] == 'streaming':
                    # 流式响应，输出增量文本
                    delta_text = response.get('text', '')
                    if delta_text:
                        print(delta_text, end='', flush=True)
                        full_text += delta_text
                    
                    # 告诉服务器继续生成
                    self.socket.send_pyobj({'status': 'continue'})
                
                elif response['status'] == 'success':
                    # 最终响应，生成完成
                    print()  # 换行
                    logger.info("Generation completed")
                    return full_text
                
                else:
                    logger.warning(f"Unknown response status: {response['status']}")
                    self.socket.send_pyobj({'status': 'continue'})
        
        except Exception as e:
            logger.error(f"Error in generate_from_messages: {str(e)}")
            return None

    def generate(self, input_ids, attention_mask, max_new_tokens=50, temperature=0.7, top_p=0.9, do_sample=True):
        input_data = {
            'input_ids': input_ids.numpy().tolist(),
            'attention_mask': attention_mask.numpy().tolist(),
            'max_new_tokens': max_new_tokens,
            'temperature': temperature,
            'top_p': top_p,
            'do_sample': do_sample
        }
        message = {'command': 'generate', 'data': input_data}
        
        logger.info("Sending generate request...")
        self.socket.send_pyobj(message)
        
        # 处理流式输出
        full_text = ""
        tokens = []
        
        while True:
            response = self.socket.recv_pyobj()
            
            if response['status'] == 'error':
                logger.error(f"Generation error: {response.get('error')}")
                return None
            
            if response['status'] == 'streaming':
                text_chunk = response.get('text', '')
                token_chunk = response.get('tokens', [])
                tokens.extend(token_chunk)
                
                print(text_chunk, end='', flush=True)
                full_text += text_chunk
                
                self.socket.send_pyobj({'status': 'received'})
            
            if response['status'] == 'success':
                output_ids = response.get('output_ids', [])[0]
                logger.info(f"Generation completed, output length: {len(output_ids)}")
                return torch.tensor([output_ids])
        
    def shutdown(self):
        """关闭连接并释放资源"""
        try:
            logger.info("Sending shutdown command...")
            self.socket.send_pyobj({'command': 'shutdown'})
            response = self.socket.recv_pyobj()
            logger.info(f"Shutdown response: {response}")
        except Exception as e:
            logger.error(f"Error during shutdown: {str(e)}")
        finally:
            self.socket.close()
            self.context.term()

def main():
    # 设置输入和输出的编码
    default_encoding = locale.getpreferredencoding()
    if default_encoding.lower() == 'ascii':
        if platform.system() == 'Windows':
            default_encoding = 'utf-8'
        else:
            default_encoding = locale.getlocale()[1]
    if not default_encoding:
        default_encoding = 'utf-8'
    
    # 设置标准输入输出的编码
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
    sys.stdin = open(sys.stdin.fileno(), mode='r', encoding=default_encoding, buffering=1)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--gguf_path', type=str, required=True)
    parser.add_argument('--worker_address', type=str, default=None,
                      help='Address of worker for distributed mode (e.g., "10.21.22.206:29500")')
    parser.add_argument('--scheduler_address', type=str, default=None,
                      help='Address of scheduler for distributed mode (e.g., "10.21.22.100:29400")')
    parser.add_argument('--max_new_tokens', type=int, default=300)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_p', type=float, default=0.9)
    args = parser.parse_args()

    if args.scheduler_address or args.worker_address:
        # 分布式模式
        logger.info("Running in distributed mode")
        model = DistributedModel(args.worker_address, args.scheduler_address)
        model.load_model(args.model_path, args.gguf_path)
        
        # 当使用调度器时，无需在客户端加载tokenizer
        tokenizer = None
        if args.worker_address and not args.scheduler_address:
            tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        
        try:
            while True:
                content = input("Chat: ")
                if content.lower() in ['quit', 'exit']:
                    logger.info("Shutting down...")
                    model.shutdown()
                    break
                elif content == "":
                    content = "Please write a piece of quicksort code in C++."
                elif os.path.isfile(content):
                    logger.info(f"Reading from file: {content}")
                    with open(content, "r", encoding='utf-8') as f:
                        content = f.read()
                
                logger.info("Creating chat messages...")
                messages = [{"role": "user", "content": content}]
                
                logger.info("Starting chat...")
                if args.scheduler_address:
                    # 使用调度器进行预处理的方法
                    outputs = model.generate_from_messages(
                        messages,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        do_sample=True
                    )
                else:
                    # 直接连接worker的原始方法
                    logger.info("Tokenizing input...")
                    input_tensor = tokenizer.apply_chat_template(
                        messages, add_generation_prompt=True, return_tensors="pt"
                    )
                    
                    logger.info("Starting generation...")
                    outputs = model.generate(
                        input_tensor,
                        torch.ones_like(input_tensor),
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        do_sample=True
                    )
                
                if outputs is not None:
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
