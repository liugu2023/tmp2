import zmq
import threading
import queue
import logging
from typing import Dict, Any, List
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Scheduler:
    def __init__(self, worker_addresses: List[str]):
        self.context = zmq.Context()
        self.task_queue = queue.Queue()
        self.worker_stats = {}  # 记录每个 worker 的状态
        self.workers = {}  # worker 连接池
        self.lock = threading.Lock()
        
        # 初始化与所有 worker 的连接
        for addr in worker_addresses:
            socket = self.context.socket(zmq.REQ)
            socket.connect(f"tcp://{addr}")
            self.workers[addr] = {
                'socket': socket,
                'busy': False,
                'last_task_time': 0,
                'total_tasks': 0,
                'avg_time': 0
            }
            logger.info(f"Connected to worker at {addr}")
    
    def get_best_worker(self) -> str:
        """选择最合适的 worker"""
        with self.lock:
            available_workers = [
                addr for addr, info in self.workers.items() 
                if not info['busy']
            ]
            if not available_workers:
                return None
            
            # 根据历史性能选择最快的 worker
            return min(
                available_workers,
                key=lambda addr: self.workers[addr]['avg_time']
            )
    
    def submit_task(self, task: Dict[str, Any]):
        """提交任务到队列"""
        self.task_queue.put(task)
        logger.info(f"Task queued. Queue size: {self.task_queue.qsize()}")
    
    def process_tasks(self):
        """处理任务队列"""
        while True:
            try:
                # 获取任务
                task = self.task_queue.get()
                
                # 等待可用的 worker
                worker_addr = None
                while worker_addr is None:
                    worker_addr = self.get_best_worker()
                    if worker_addr is None:
                        time.sleep(0.1)
                
                # 分配任务给 worker
                worker = self.workers[worker_addr]
                worker['busy'] = True
                start_time = time.time()
                
                try:
                    # 发送任务
                    worker['socket'].send_pyobj({
                        'command': 'generate',
                        'data': task
                    })
                    
                    # 获取结果
                    result = worker['socket'].recv_pyobj()
                    
                    # 更新 worker 统计信息
                    end_time = time.time()
                    task_time = end_time - start_time
                    with self.lock:
                        worker['total_tasks'] += 1
                        worker['avg_time'] = (
                            (worker['avg_time'] * (worker['total_tasks'] - 1) + task_time)
                            / worker['total_tasks']
                        )
                        worker['last_task_time'] = end_time
                        worker['busy'] = False
                    
                    logger.info(f"Task completed by worker {worker_addr}")
                    
                except Exception as e:
                    logger.error(f"Error processing task on worker {worker_addr}: {e}")
                    worker['busy'] = False
                    # 重新入队
                    self.task_queue.put(task)
                
            except Exception as e:
                logger.error(f"Error in task processing loop: {e}")
                time.sleep(1)
    
    def start(self):
        """启动调度器"""
        # 启动任务处理线程
        self.process_thread = threading.Thread(target=self.process_tasks)
        self.process_thread.daemon = True
        self.process_thread.start()
        logger.info("Scheduler started")
    
    def shutdown(self):
        """关闭调度器"""
        logger.info("Shutting down scheduler...")
        # 关闭所有 worker 连接
        for addr, worker in self.workers.items():
            try:
                worker['socket'].send_pyobj({'command': 'shutdown'})
                worker['socket'].recv_pyobj()  # 等待确认
                worker['socket'].close()
            except Exception as e:
                logger.error(f"Error shutting down worker {addr}: {e}")
        
        self.context.term()
        logger.info("Scheduler shutdown complete") 