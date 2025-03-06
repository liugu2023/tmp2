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
        
    def load_model_on_workers(self, model_path: str, gguf_path: str) -> Dict[str, Any]:
        """加载模型到所有worker上"""
        logger.info(f"加载模型: {model_path}, GGUF: {gguf_path}")
        
        # 首先加载到KV缓存服务器
        logger.info("发送模型加载命令到KV缓存服务器...")
        self.kvcache_socket.send_pyobj({
            'command': 'load_model',
            'model_path': model_path,
            'gguf_path': gguf_path
        })
        
        response = self.kvcache_socket.recv_pyobj()
        if response.get('status') != 'success':
            logger.error(f"KV缓存服务器加载模型失败: {response.get('error')}")
            return {'status': 'error', 'error': f"KV缓存服务器加载模型失败: {response.get('error')}"}
        
        # 然后加载到各个worker
        logger.info("发送模型加载命令到所有workers...")
        success = True
        errors = []
        
        for addr, socket in self.workers:
            try:
                logger.info(f"加载模型到worker: {addr}")
                socket.send_pyobj({
                    'command': 'load_model',
                    'model_path': model_path,
                    'gguf_path': gguf_path
                })
                
                worker_response = socket.recv_pyobj()
                if worker_response.get('status') != 'success':
                    logger.error(f"Worker {addr} 加载模型失败: {worker_response.get('error')}")
                    success = False
                    errors.append(f"Worker {addr}: {worker_response.get('error')}")
            except Exception as e:
                logger.error(f"与Worker {addr}通信失败: {str(e)}")
                success = False
                errors.append(f"Worker {addr}: 通信失败 - {str(e)}")
        
        # 返回结果
        if success:
            logger.info("所有worker成功加载模型")
            return {'status': 'success', 'message': '所有worker成功加载模型'}
        else:
            error_msg = "; ".join(errors)
            logger.error(f"部分或全部worker加载模型失败: {error_msg}")
            return {'status': 'error', 'error': f"加载模型失败: {error_msg}"}
    
    def generate(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理生成请求"""
        try:
            # 确保max_length是整数
            if 'max_length' in input_data:
                max_length = input_data['max_length']
                if isinstance(max_length, (list, tuple)):
                    max_length = max_length[0]
                input_data['max_length'] = int(max_length)
            
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
        
        except Exception as e:
            logger.error(f"生成过程中出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'status': 'error',
                'error': str(e)
            }
    
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
    
    def shutdown_workers(self):
        """关闭所有连接的worker进程"""
        logger.info("正在关闭所有worker进程...")
        
        for addr, socket in self.workers:
            try:
                logger.info(f"正在关闭worker: {addr}")
                socket.send_pyobj({'command': 'shutdown'})
                response = socket.recv_pyobj()  # 等待确认
                logger.info(f"Worker {addr} 回应: {response.get('message', 'No message')}")
            except Exception as e:
                logger.error(f"关闭worker {addr}时出错: {str(e)}")
        
        # 关闭KV缓存服务器
        try:
            logger.info("正在关闭KV缓存服务器...")
            self.kvcache_socket.send_pyobj({'command': 'shutdown'})
            response = self.kvcache_socket.recv_pyobj()
            logger.info(f"KV缓存服务器回应: {response.get('message', 'No message')}")
        except Exception as e:
            logger.error(f"关闭KV缓存服务器时出错: {str(e)}")
        
        logger.info("所有worker进程已关闭")
        return True
    
    def process_client_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """处理客户端消息"""
        try:
            command = message.get('command', '')
            logger.info(f"Received command: {command}")
            
            if command == 'load_model':
                # 从message直接获取参数，而不是使用message.get
                # 这样如果缺少必要参数，会立即抛出KeyError
                model_path = message['model_path']
                gguf_path = message.get('gguf_path', '')
                
                # 验证model_path不为空
                if not model_path:
                    return {'status': 'error', 'error': '模型路径不能为空'}
                
                # 打印接收到的路径，以便调试
                logger.info(f"从客户端收到路径 - model_path: '{model_path}', gguf_path: '{gguf_path}'")
                
                return self.load_model_on_workers(model_path, gguf_path)
            elif command == 'generate':
                return self.generate(message.get('data', {}))
            elif command == 'shutdown':
                # 关闭所有worker进程
                result = self.shutdown_workers()
                return {'status': 'success' if result else 'error', 'message': 'Shutdown complete'}
            elif command == 'ping':
                return {'status': 'success', 'message': 'pong'}
            else:
                return {
                    'status': 'error',
                    'error': f'Unknown command: {command}'
                }
        except KeyError as e:
            # 处理缺少必要参数的情况
            logger.error(f"消息缺少必要参数: {str(e)}")
            return {'status': 'error', 'error': f'缺少必要参数: {str(e)}'}
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def run(self):
        """主服务循环"""
        logger.info("Central server is running and ready to accept requests")
        
        try:
            while True:
                # 接收客户端请求
                message = self.client_socket.recv_pyobj()
                result = self.process_client_message(message)
                self.client_socket.send_pyobj(result)
        
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