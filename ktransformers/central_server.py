import zmq
import torch
import logging
import time
import argparse
from typing import Dict, Any, List
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CentralServer:
    def __init__(self, listen_port: int, worker_addresses: List[str], kvcache_address: str):
        """
        初始化中央服务器
        
        Args:
            listen_port: 监听客户端请求的端口
            worker_addresses: 所有worker的地址列表，如 ["10.21.22.206:29500", "10.21.22.206:29501", ...]
            kvcache_address: KV缓存服务器地址，如 "10.21.22.206:29600"
        """
        # 设置客户端Socket
        self.context = zmq.Context()
        self.client_socket = self.context.socket(zmq.REP)
        self.client_socket.bind(f"tcp://0.0.0.0:{listen_port}")
        logger.info(f"Central server listening for clients on port {listen_port}")
        
        # 连接到各个worker
        self.workers = []
        for addr in worker_addresses:
            worker_socket = self.context.socket(zmq.REQ)
            worker_socket.connect(f"tcp://{addr}")
            self.workers.append((addr, worker_socket))
            logger.info(f"Connected to worker at {addr}")
        
        # 连接到KV缓存服务器
        self.kvcache_socket = self.context.socket(zmq.REQ)
        self.kvcache_socket.connect(f"tcp://{kvcache_address}")
        logger.info(f"Connected to KV Cache server at {kvcache_address}")
        
        # 活跃的批处理任务
        self.active_batches = {}
        
    def load_model_on_workers(self, model_path: str, gguf_path: str):
        """在所有worker上加载模型"""
        logger.info(f"Loading model on all workers: {model_path}")
        
        # 首先通知KVCache服务器加载模型
        try:
            logger.info("通知KVCache服务器加载模型...")
            self.kvcache_socket.send_pyobj({
                'command': 'load_model',
                'model_path': model_path,
                'gguf_path': gguf_path
            })
            response = self.kvcache_socket.recv_pyobj()
            if response['status'] != 'success':
                logger.error(f"KVCache服务器加载模型失败: {response.get('error')}")
                return False
        except Exception as e:
            logger.error(f"与KVCache服务器通信出错: {str(e)}")
            return False
        
        # 然后通知所有worker
        success = True
        for addr, socket in self.workers:
            try:
                logger.info(f"Loading model on worker at {addr}...")
                socket.send_pyobj({
                    'command': 'load_model',
                    'model_path': model_path,
                    'gguf_path': gguf_path
                })
                response = socket.recv_pyobj()
                if response['status'] != 'success':
                    logger.error(f"Failed to load model on worker at {addr}: {response.get('error')}")
                    success = False
            except Exception as e:
                logger.error(f"Error communicating with worker at {addr}: {str(e)}")
                success = False
        
        return success
    
    def generate(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理生成请求"""
        batch_id = f"batch_{time.time()}"
        self.active_batches[batch_id] = {
            'worker_responses': [None] * len(self.workers),
            'completed': False,
            'result': None,
            'streaming_text': ""
        }
        
        # 向所有worker发送生成请求
        for i, (addr, socket) in enumerate(self.workers):
            message = {
                'command': 'generate',
                'data': {
                    **input_data,
                    'batch_id': batch_id
                }
            }
            socket.send_pyobj(message)
            
            # 启动线程接收流式输出
            thread = threading.Thread(
                target=self._handle_worker_response,
                args=(i, addr, socket, batch_id)
            )
            thread.daemon = True
            thread.start()
        
        # 等待所有worker完成
        while not self.active_batches[batch_id]['completed']:
            # 发送流式响应给客户端
            current_text = self.active_batches[batch_id]['streaming_text']
            if current_text:
                self.client_socket.send_pyobj({
                    'status': 'streaming',
                    'text': current_text,
                    'finished': False
                })
                # 等待客户端确认
                self.client_socket.recv_pyobj()
            time.sleep(0.1)
        
        # 清理KV缓存
        self.kvcache_socket.send_pyobj({
            'command': 'clear',
            'batch_id': batch_id
        })
        self.kvcache_socket.recv_pyobj()
        
        # 发送最终结果
        result = self.active_batches[batch_id]['result']
        del self.active_batches[batch_id]
        
        return result
    
    def _handle_worker_response(self, worker_idx: int, addr: str, socket: zmq.Socket, batch_id: str):
        """处理单个worker的响应"""
        try:
            response = socket.recv_pyobj()
            
            # 处理流式响应
            if response['status'] == 'streaming':
                self.active_batches[batch_id]['streaming_text'] = response['text']
                socket.send_pyobj({'status': 'received'})
                
                # 如果流式生成未结束，递归继续接收
                if not response.get('finished', False):
                    self._handle_worker_response(worker_idx, addr, socket, batch_id)
                    return
            
            # 处理最终响应
            if response['status'] == 'success':
                self.active_batches[batch_id]['worker_responses'][worker_idx] = response
                
                # 检查是否所有worker都完成了
                responses = self.active_batches[batch_id]['worker_responses']
                if all(r is not None for r in responses):
                    # 合并所有worker的结果
                    self.active_batches[batch_id]['result'] = responses[0]  # 使用第一个worker的结果
                    self.active_batches[batch_id]['completed'] = True
            
            elif response['status'] == 'error':
                logger.error(f"Error from worker {addr}: {response.get('error')}")
                # 标记任务失败
                self.active_batches[batch_id]['result'] = {
                    'status': 'error',
                    'error': f"Worker {addr} error: {response.get('error')}"
                }
                self.active_batches[batch_id]['completed'] = True
                
        except Exception as e:
            logger.error(f"Error handling response from {addr}: {str(e)}")
            self.active_batches[batch_id]['result'] = {
                'status': 'error',
                'error': f"Communication error with worker {addr}: {str(e)}"
            }
            self.active_batches[batch_id]['completed'] = True
    
    def run(self):
        """主服务循环"""
        logger.info("Central server is running and ready to accept requests")
        
        try:
            while True:
                # 接收客户端请求
                message = self.client_socket.recv_pyobj()
                command = message.get('command')
                
                if command == 'load_model':
                    success = self.load_model_on_workers(
                        message['model_path'],
                        message['gguf_path']
                    )
                    self.client_socket.send_pyobj({
                        'status': 'success' if success else 'error'
                    })
                
                elif command == 'generate':
                    result = self.generate(message['data'])
                    self.client_socket.send_pyobj(result)
                
                elif command == 'shutdown':
                    logger.info("Shutting down central server and all workers")
                    
                    # 关闭所有worker
                    for addr, socket in self.workers:
                        try:
                            socket.send_pyobj({'command': 'shutdown'})
                            socket.recv_pyobj()
                        except Exception as e:
                            logger.error(f"Error shutting down worker {addr}: {str(e)}")
                    
                    # 关闭KV Cache服务器
                    try:
                        self.kvcache_socket.send_pyobj({'command': 'shutdown'})
                        self.kvcache_socket.recv_pyobj()
                    except Exception as e:
                        logger.error(f"Error shutting down KV Cache server: {str(e)}")
                    
                    self.client_socket.send_pyobj({'status': 'shutdown'})
                    break
                
                else:
                    logger.error(f"Unknown command: {command}")
                    self.client_socket.send_pyobj({
                        'status': 'error',
                        'error': f"Unknown command: {command}"
                    })
        
        finally:
            logger.info("Cleaning up central server resources")
            self.client_socket.close()
            for _, socket in self.workers:
                socket.close()
            self.kvcache_socket.close()
            self.context.term()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=29400)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--worker_base_port', type=int, default=29500)
    parser.add_argument('--kvcache_port', type=int, default=29600)
    parser.add_argument('--worker_host', type=str, default='10.21.22.206')
    args = parser.parse_args()
    
    # 构建worker地址列表
    worker_addresses = [
        f"{args.worker_host}:{args.worker_base_port + i}" 
        for i in range(args.num_workers)
    ]
    
    # 创建并运行中央服务器
    server = CentralServer(
        args.port, 
        worker_addresses,
        f"{args.worker_host}:{args.kvcache_port}"
    )
    server.run()

if __name__ == '__main__':
    main() 