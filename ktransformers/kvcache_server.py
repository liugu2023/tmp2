"""
KV Cache 集中式服务器
负责存储和同步多个 worker 之间的 KV Cache
"""

import zmq
import torch
import logging
import numpy as np
import time
from typing import Dict, Any, List
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KVCacheServer:
    def __init__(self, server_address: str, num_workers: int = 8):
        self.num_workers = num_workers
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://{server_address}")
        
        # KV Cache 存储
        self.kv_caches = {}
        self.cache_lock = threading.Lock()
        
        # 性能统计
        self.stats = {
            "total_requests": 0,
            "store_requests": 0,
            "retrieve_requests": 0,
            "total_data_bytes": 0,
            "start_time": time.time()
        }
        
        logger.info(f"KV Cache Server started at {server_address}")
        logger.info(f"Configured for {num_workers} workers")
    
    def run(self):
        """运行服务器主循环"""
        logger.info("KV Cache Server ready to receive requests")
        try:
            while True:
                # 接收请求
                request = self.socket.recv_pyobj()
                self.stats["total_requests"] += 1
                
                # 处理不同类型的请求
                command = request.get("command")
                if command == "store":
                    response = self._handle_store(request)
                elif command == "retrieve":
                    response = self._handle_retrieve(request)
                elif command == "clear":
                    response = self._handle_clear(request)
                elif command == "stats":
                    response = self._handle_stats()
                elif command == "shutdown":
                    response = {"status": "shutdown"}
                    self.socket.send_pyobj(response)
                    break
                else:
                    response = {"status": "error", "error": f"Unknown command: {command}"}
                
                # 发送响应
                self.socket.send_pyobj(response)
                
                # 记录数据大小
                if "data_size" in request:
                    self.stats["total_data_bytes"] += request["data_size"]
        finally:
            logger.info("Shutting down KV Cache Server...")
            self.socket.close()
            self.context.term()
    
    def _handle_store(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """处理存储 KV Cache 的请求"""
        self.stats["store_requests"] += 1
        
        request_id = request["request_id"]
        worker_id = request["worker_id"]
        layer_id = request["layer_id"]
        kv_tensor = request["kv_tensor"]
        
        # 使用锁确保线程安全
        with self.cache_lock:
            # 如果这个请求ID还没有缓存，创建一个新的
            if request_id not in self.kv_caches:
                self.kv_caches[request_id] = {}
            
            # 存储KV缓存
            key = f"worker_{worker_id}_layer_{layer_id}"
            self.kv_caches[request_id][key] = kv_tensor
            logger.debug(f"Stored KV cache for request {request_id}, worker {worker_id}, layer {layer_id}")
        
        return {"status": "success"}
    
    def _handle_retrieve(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """处理检索 KV Cache 的请求"""
        self.stats["retrieve_requests"] += 1
        
        request_id = request["request_id"]
        worker_id = request["worker_id"]
        layer_id = request["layer_id"]
        
        # 使用锁确保线程安全
        with self.cache_lock:
            # 检查请求ID是否存在
            if request_id not in self.kv_caches:
                return {"status": "error", "error": f"Request ID {request_id} not found"}
            
            # 检索KV缓存
            key = f"worker_{worker_id}_layer_{layer_id}"
            if key not in self.kv_caches[request_id]:
                return {"status": "error", "error": f"KV cache for {key} not found"}
            
            kv_tensor = self.kv_caches[request_id][key]
            logger.debug(f"Retrieved KV cache for request {request_id}, worker {worker_id}, layer {layer_id}")
        
        return {"status": "success", "kv_tensor": kv_tensor}
    
    def _handle_clear(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """清除指定请求ID的KV缓存"""
        request_id = request["request_id"]
        
        # 使用锁确保线程安全
        with self.cache_lock:
            if request_id in self.kv_caches:
                del self.kv_caches[request_id]
                logger.info(f"Cleared KV cache for request {request_id}")
            else:
                logger.warning(f"Request ID {request_id} not found for clearing")
        
        # 执行垃圾收集
        if len(self.kv_caches) % 10 == 0:  # 每10个请求清理一次
            torch.cuda.empty_cache()
        
        return {"status": "success"}
    
    def _handle_stats(self) -> Dict[str, Any]:
        """返回服务器统计信息"""
        current_time = time.time()
        uptime = current_time - self.stats["start_time"]
        
        stats = {
            "status": "success",
            "total_requests": self.stats["total_requests"],
            "store_requests": self.stats["store_requests"],
            "retrieve_requests": self.stats["retrieve_requests"],
            "total_data_bytes": self.stats["total_data_bytes"],
            "uptime": uptime,
            "requests_per_second": self.stats["total_requests"] / uptime if uptime > 0 else 0,
            "active_caches": len(self.kv_caches),
            "memory_usage": {
                "allocated": torch.cuda.memory_allocated() / 1024**3,
                "reserved": torch.cuda.memory_reserved() / 1024**3
            }
        }
        
        return stats

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--server_address', type=str, default="10.21.22.206:29600")
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()
    
    server = KVCacheServer(args.server_address, args.num_workers)
    server.run()

if __name__ == "__main__":
    main() 