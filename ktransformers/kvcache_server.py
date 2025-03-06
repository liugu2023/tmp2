import zmq
import torch
import logging
from typing import Dict, Any
import numpy as np
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KVCacheServer:
    def __init__(self, port: int = 29600, max_retries: int = 5, model_path: str = None, gguf_path: str = None):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        
        # 尝试绑定端口，如果失败则重试
        self.port = port
        retry_count = 0
        bound = False
        
        while not bound and retry_count < max_retries:
            try:
                self.socket.bind(f"tcp://0.0.0.0:{self.port}")
                bound = True
                logger.info(f"KV Cache Server 成功绑定到端口 {self.port}")
            except zmq.error.ZMQError as e:
                retry_count += 1
                logger.warning(f"端口 {self.port} 被占用，尝试重新绑定... ({retry_count}/{max_retries})")
                if retry_count >= max_retries:
                    logger.error(f"无法绑定到端口 {self.port}，请手动终止占用该端口的进程")
                    raise
                time.sleep(1)
        
        # 存储 KV Cache
        # {batch_id: {layer_id: {'k': tensor, 'v': tensor}}}
        self.cache = {}
        logger.info(f"KV Cache Server started on port {port}")
        
        # 如果提供了模型路径，初始化模型加载器
        self.model_loader = None
        if model_path:
            from ktransformers.model_loader import ModelLoader
            logger.info(f"初始化模型加载器，模型路径: {model_path}")
            self.model_loader = ModelLoader(model_path, gguf_path)
            self.model_loader.load_model()
            logger.info("模型加载器初始化完成")
    
    def run(self):
        while True:
            try:
                message = self.socket.recv_pyobj()
                command = message['command']
                
                if command == 'store':
                    # 存储 KV Cache
                    batch_id = message['batch_id']
                    layer_id = message['layer_id']
                    k = message['k']
                    v = message['v']
                    
                    if batch_id not in self.cache:
                        self.cache[batch_id] = {}
                    
                    self.cache[batch_id][layer_id] = {
                        'k': k.cpu(),  # 存储在 CPU 内存中
                        'v': v.cpu()
                    }
                    
                    self.socket.send_pyobj({'status': 'success'})
                
                elif command == 'retrieve':
                    # 获取 KV Cache
                    batch_id = message['batch_id']
                    layer_id = message['layer_id']
                    device = message['device']
                    
                    if batch_id not in self.cache or layer_id not in self.cache[batch_id]:
                        self.socket.send_pyobj({
                            'status': 'error',
                            'error': 'Cache not found'
                        })
                        continue
                    
                    k = self.cache[batch_id][layer_id]['k'].to(device)
                    v = self.cache[batch_id][layer_id]['v'].to(device)
                    
                    self.socket.send_pyobj({
                        'status': 'success',
                        'k': k,
                        'v': v
                    })
                
                elif command == 'clear':
                    # 清除指定 batch 的缓存
                    batch_id = message['batch_id']
                    if batch_id in self.cache:
                        del self.cache[batch_id]
                    self.socket.send_pyobj({'status': 'success'})
                
                elif command == 'shutdown':
                    logger.info("Shutting down KV Cache Server...")
                    self.socket.send_pyobj({'status': 'success'})
                    break
                
                # 添加模型参数相关命令
                elif command == 'get_config':
                    if self.model_loader:
                        self.socket.send_pyobj({
                            'status': 'success',
                            'config': self.model_loader.get_config()
                        })
                    else:
                        self.socket.send_pyobj({
                            'status': 'error',
                            'error': 'Model loader not initialized'
                        })
                        
                elif command == 'get_tokenizer':
                    if self.model_loader:
                        self.socket.send_pyobj({
                            'status': 'success',
                            'tokenizer': self.model_loader.get_tokenizer()
                        })
                    else:
                        self.socket.send_pyobj({
                            'status': 'error',
                            'error': 'Model loader not initialized'
                        })
                        
                elif command == 'get_layer_params':
                    layer_index = message['layer_index']
                    if self.model_loader:
                        params = self.model_loader.get_layer_params(layer_index)
                        self.socket.send_pyobj({
                            'status': 'success' if params is not None else 'error',
                            'params': params
                        })
                    else:
                        self.socket.send_pyobj({
                            'status': 'error',
                            'error': 'Model loader not initialized'
                        })
                        
                elif command == 'get_shared_params':
                    if self.model_loader:
                        self.socket.send_pyobj({
                            'status': 'success',
                            'params': self.model_loader.get_shared_params()
                        })
                    else:
                        self.socket.send_pyobj({
                            'status': 'error',
                            'error': 'Model loader not initialized'
                        })
                
            except Exception as e:
                logger.error(f"Error in KV Cache Server: {str(e)}")
                self.socket.send_pyobj({
                    'status': 'error',
                    'error': str(e)
                })
        
        self.socket.close()
        self.context.term()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=29600)
    args = parser.parse_args()
    
    server = KVCacheServer(args.port)
    server.run()

if __name__ == '__main__':
    main() 