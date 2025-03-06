import zmq
import torch
import logging
from typing import Dict, Any
import numpy as np
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KVCacheServer:
    def __init__(self, port: int = 29600, max_retries: int = 5):
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