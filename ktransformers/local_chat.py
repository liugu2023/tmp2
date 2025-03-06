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
from typing import Dict

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
    """分布式模型客户端类"""
    
    def __init__(self, server_address):
        """初始化分布式模型客户端
        
        Args:
            server_address: 中央服务器地址，如 "10.21.22.206:29400"
        """
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{server_address}")
        self.distributed = True
        
        # 初始化tokenizer属性
        self.tokenizer = None
        
        logger.info(f"Connected to central server at {server_address}")
    
    def load_model(self, model_path: str, gguf_path: str = None) -> bool:
        """加载模型"""
        print(f"正在加载模型: {model_path}")
        if gguf_path:
            print(f"使用GGUF模型: {gguf_path}")
        
        # 确保路径不为空字符串
        if not model_path or model_path.strip() == '':
            print("错误: 模型路径不能为空")
            return False
        
        # 打印详细的请求内容，用于调试
        request_data = {
            'model_path': model_path,
            'gguf_path': gguf_path
        }
        print(f"发送加载模型请求: {request_data}")
        
        response = self._send_command('load_model', request_data)
        
        # 检查response是否是字典
        if isinstance(response, bool):
            success = response
            error = "未知错误" if not success else None
        else:
            success = response.get('status') == 'success'
            error = response.get('error')
        
        if not success:
            print(f"模型加载失败: {error}")
            return False
        
        # 在成功加载模型后，初始化tokenizer
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            print(f"Tokenizer loaded from {model_path}")
        except Exception as e:
            print(f"无法加载tokenizer: {e}")
            print("将使用默认输出显示")
            self.tokenizer = None
        
        print("模型加载成功!")
        return True

    def _send_command(self, command: str, data: Dict = None) -> Dict:
        """发送命令到服务器并接收响应"""
        if data is None:
            data = {}
        
        message = {
            'command': command,
            'data': data
        }
        
        try:
            self.socket.send_pyobj(message)
            response = self.socket.recv_pyobj()
            
            # 确保返回字典
            if not isinstance(response, dict):
                return {'status': 'success' if response else 'error', 'value': response}
            
            return response
        except Exception as e:
            print(f"通信错误: {str(e)}")
            return {'status': 'error', 'error': str(e)}

    def generate(self, input_ids, attention_mask, **kwargs):
        """生成文本"""
        try:
            # 准备请求数据
            generate_data = {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }
            # 添加其他生成参数
            generate_data.update(kwargs)
            
            # 发送生成请求
            response = self._send_command('generate', generate_data)
            
            if response.get('status') != 'success':
                error_msg = response.get('error', 'Unknown error')
                print(f"生成失败: {error_msg}")
                return []
            
            # 获取输出token IDs
            output_ids = response.get('output_ids', [])
            
            print("生成成功!")
            # 修复: 使用长度检查而不是直接用张量作为布尔值
            if isinstance(output_ids, torch.Tensor):
                has_outputs = len(output_ids) > 0
            else:
                has_outputs = bool(output_ids)
            
            if has_outputs:
                # 确保输出为列表，以便切片操作安全
                if isinstance(output_ids, torch.Tensor):
                    # 修复: 确保转换为一维列表
                    output_ids_list = output_ids.tolist()
                    if isinstance(output_ids_list, list) and isinstance(output_ids_list[0], list):
                        # 处理嵌套列表情况 [[id1, id2, ...]]
                        output_ids_list = output_ids_list[0]
                    output_preview = output_ids_list[:10]
                else:
                    # 处理已经是列表的情况
                    output_ids_list = output_ids
                    if isinstance(output_ids_list, list) and len(output_ids_list) > 0 and isinstance(output_ids_list[0], list):
                        # 处理嵌套列表
                        output_ids_list = output_ids_list[0]
                    output_preview = output_ids_list[:10]
                
                print(f"生成的tokens: {output_preview}...")
                
                # 安全地检查tokenizer属性
                tokenizer = getattr(self, 'tokenizer', None)
                if tokenizer is not None:
                    try:
                        # 使用已展平的一维列表进行解码
                        output_text = tokenizer.decode(output_ids_list)
                        print(f"\n生成的文本:\n{output_text}\n")
                    except Exception as e:
                        print(f"\n解码文本时出错: {str(e)}")
                        print(f"Token IDs: {output_ids_list}\n")
                else:
                    print("\n无法解码文本: tokenizer不可用\n")
            else:
                print("警告: 未收到任何生成的tokens")
            
            return output_ids
        
        except Exception as e:
            print(f"生成过程中出错: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return []

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
    parser.add_argument('--server_address', type=str, default=None,
                      help='Address of central server (e.g., "10.21.22.206:29400")')
    parser.add_argument('--max_new_tokens', type=int, default=300)
    args = parser.parse_args()

    if args.server_address:
        # 分布式模式
        logger.info("Running in distributed mode")
        model = DistributedModel(args.server_address)
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
