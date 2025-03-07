"""
任务调度服务器
负责预处理输入数据并转发客户端请求到计算节点
"""

import zmq
import logging
import time
import argparse
import torch
from typing import Dict, Any, List
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SchedulerServer:
    def __init__(self, server_address: str, compute_address: str, model_path: str):
        # 初始化ZMQ上下文
        self.context = zmq.Context()
        
        # 设置面向客户端的套接字
        self.client_socket = self.context.socket(zmq.REP)
        self.client_socket.bind(f"tcp://{server_address}")
        logger.info(f"Scheduler listening for clients at {server_address}")
        
        # 设置面向计算节点的套接字
        self.compute_socket = self.context.socket(zmq.REQ)
        self.compute_address = compute_address
        self.compute_socket.connect(f"tcp://{compute_address}")
        logger.info(f"Connected to compute node at {compute_address}")
        
        # 加载tokenizer
        self.model_path = model_path
        logger.info(f"Loading tokenizer from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        logger.info(f"Tokenizer loaded successfully")
        
        # 统计信息
        self.stats = {
            "total_requests": 0,
            "start_time": time.time(),
            "active_sessions": 0,
            "total_tokens_processed": 0
        }
    
    def run(self):
        """运行调度器主循环"""
        logger.info("Scheduler server ready to receive requests")
        try:
            while True:
                # 接收来自客户端的请求
                client_request = self.client_socket.recv_pyobj()
                self.stats["total_requests"] += 1
                
                command = client_request.get("command")
                logger.info(f"Received {command} request from client")
                
                if command == "shutdown":
                    # 处理关闭命令
                    self.client_socket.send_pyobj({"status": "shutdown"})
                    break
                elif command == "stats":
                    # 返回统计信息
                    response = self._get_stats()
                    self.client_socket.send_pyobj(response)
                elif command == "chat":
                    # 预处理聊天输入并转发
                    processed_request = self._preprocess_chat(client_request)
                    self._forward_to_compute(processed_request)
                else:
                    # 转发所有其他命令到计算节点
                    try:
                        self._forward_to_compute(client_request)
                    except Exception as e:
                        logger.error(f"Error forwarding request: {str(e)}")
                        self.client_socket.send_pyobj({
                            "status": "error",
                            "error": f"Scheduler error: {str(e)}"
                        })
        finally:
            logger.info("Shutting down scheduler...")
            self.client_socket.close()
            self.compute_socket.close()
            self.context.term()
    
    def _preprocess_chat(self, client_request: Dict[str, Any]) -> Dict[str, Any]:
        """预处理聊天输入，将文本转换为token"""
        messages = client_request.get("messages", [])
        max_new_tokens = client_request.get("max_new_tokens", 512)
        temperature = client_request.get("temperature", 0.7)
        top_p = client_request.get("top_p", 0.9)
        do_sample = client_request.get("do_sample", True)
        
        logger.info(f"Preprocessing chat input with {len(messages)} messages")
        
        # 使用tokenizer处理输入
        input_tensor = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
        
        # 创建注意力掩码
        attention_mask = torch.ones_like(input_tensor)
        
        # 记录token数量
        input_tokens = len(input_tensor[0])
        self.stats["total_tokens_processed"] += input_tokens
        
        logger.info(f"Input processed: {input_tokens} tokens")
        
        # 创建计算节点请求
        compute_request = {
            "command": "generate",
            "data": {
                "input_ids": input_tensor.numpy().tolist(),
                "attention_mask": attention_mask.numpy().tolist(),
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "do_sample": do_sample
            }
        }
        
        return compute_request
    
    def _forward_to_compute(self, request: Dict[str, Any]):
        """转发请求到计算节点并处理响应"""
        try:
            # 将请求转发到计算节点
            self.compute_socket.send_pyobj(request)
            
            # 如果是generate命令，则处理流式响应
            if request.get("command") == "generate":
                self._handle_streaming_response()
            else:
                # 普通命令直接转发响应
                compute_response = self.compute_socket.recv_pyobj()
                self.client_socket.send_pyobj(compute_response)
        except Exception as e:
            logger.error(f"Error in forward_to_compute: {str(e)}")
            self.client_socket.send_pyobj({
                "status": "error",
                "error": f"Scheduler error: {str(e)}"
            })
    
    def _handle_streaming_response(self):
        """处理来自计算节点的流式响应"""
        # 接收来自计算节点的流式响应并转发给客户端
        while True:
            compute_response = self.compute_socket.recv_pyobj()
            
            # 转发响应给客户端
            self.client_socket.send_pyobj(compute_response)
            
            # 如果是最终响应，退出循环
            if compute_response.get('status') == 'success':
                break
            
            # 如果是流式响应，等待客户端确认
            if compute_response.get('status') == 'streaming':
                client_ack = self.client_socket.recv_pyobj()
                self.compute_socket.send_pyobj(client_ack)
    
    def _get_stats(self) -> Dict[str, Any]:
        """获取调度器统计信息"""
        current_time = time.time()
        uptime = current_time - self.stats["start_time"]
        
        return {
            "status": "success",
            "total_requests": self.stats["total_requests"],
            "uptime": uptime,
            "active_sessions": self.stats["active_sessions"],
            "total_tokens_processed": self.stats["total_tokens_processed"],
            "compute_node": self.compute_address,
            "model_path": self.model_path
        }

def main():
    parser = argparse.ArgumentParser(description="KTransformers Scheduler Server")
    parser.add_argument('--server_address', type=str, default="0.0.0.0:29400",
                        help='Address to listen for client connections')
    parser.add_argument('--compute_address', type=str, default="10.21.22.206:29500",
                        help='Address of compute node worker')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the model for tokenization')
    args = parser.parse_args()
    
    server = SchedulerServer(args.server_address, args.compute_address, args.model_path)
    server.run()

if __name__ == "__main__":
    main() 