import os
import torch
import zmq
import pickle
from typing import Dict, Any, List, Optional, Tuple
import logging
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, GenerationConfig, TextStreamer
import time
import subprocess
import multiprocessing
import uuid
import torch.nn.functional as F
import gc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_numa_nodes():
    """获取 NUMA 节点信息"""
    try:
        output = subprocess.check_output(['numactl', '--hardware']).decode()
        nodes = []
        for line in output.split('\n'):
            if line.startswith('node ') and 'cpus:' in line:
                node_id = int(line.split()[1])
                nodes.append(node_id)
        return nodes
    except Exception as e:
        logger.error(f"Error getting NUMA nodes: {e}")
        return [0]  # 默认返回节点0

class LayerMemoryManager:
    """处理模型层的动态GPU内存管理"""
    
    def __init__(self, gpu_id: int, max_gpu_memory: str = "3GiB"):
        self.gpu_id = gpu_id
        self.max_gpu_memory = max_gpu_memory
        self.current_layers_on_gpu = set()
        self.device = f"cuda:{gpu_id}"
        
    def move_to_gpu(self, layer):
        """将层移动到GPU"""
        # 确保有足够的显存
        self._ensure_memory_available()
        
        # 检查层是否已在GPU上
        if id(layer) in self.current_layers_on_gpu:
            return
        
        # 将层移动到GPU
        layer_device = next(layer.parameters()).device
        if str(layer_device) != self.device:
            layer.to(self.device)
            self.current_layers_on_gpu.add(id(layer))
            logger.debug(f"Moved layer to GPU {self.gpu_id}, current layers on GPU: {len(self.current_layers_on_gpu)}")
    
    def move_to_cpu(self, layer):
        """将层移回CPU"""
        layer_device = next(layer.parameters()).device
        if str(layer_device) == self.device:
            layer.to("cpu")
            self.current_layers_on_gpu.discard(id(layer))
            logger.debug(f"Moved layer to CPU, current layers on GPU: {len(self.current_layers_on_gpu)}")
    
    def _ensure_memory_available(self):
        """确保有足够的GPU内存可用，必要时释放内存"""
        # 当前GPU使用的显存
        current_memory = torch.cuda.memory_allocated(self.gpu_id) / (1024**3)  # 转换为GB
        max_memory = float(self.max_gpu_memory.replace("GiB", ""))
        
        # 如果使用超过80%的最大显存，则清理一些层
        if current_memory > max_memory * 0.8:
            logger.info(f"GPU {self.gpu_id} memory usage high ({current_memory:.2f}GB/{max_memory}GB), clearing cache")
            torch.cuda.empty_cache()
            gc.collect()

class ModelWorker:
    def __init__(self, address: str, worker_id: int = 0, num_workers: int = 1, kvcache_address: str = None):
        self.worker_id = worker_id
        self.num_workers = num_workers
        self.kvcache_address = kvcache_address
        
        # 如果提供了KV Cache服务器地址，则连接
        if self.kvcache_address:
            self.kvcache_context = zmq.Context()
            self.kvcache_socket = self.kvcache_context.socket(zmq.REQ)
            self.kvcache_socket.connect(f"tcp://{kvcache_address}")
            logger.info(f"Worker {worker_id} connected to KV Cache Server at {kvcache_address}")
        else:
            self.kvcache_socket = None
            logger.info(f"Worker {worker_id} running without KV Cache Server")
        
        # 设置 GPU 设备
        if torch.cuda.is_available():
            # 根据 worker_id 分配 GPU
            self.gpu_id = 0 if worker_id < 4 else 1
            torch.cuda.set_device(self.gpu_id)
            logger.info(f"Worker {worker_id} using GPU {self.gpu_id}")
            
            # 创建内存管理器
            self.memory_manager = LayerMemoryManager(self.gpu_id, "3GiB")
        else:
            self.gpu_id = None
            logger.warning("CUDA not available, using CPU only.")
        
        # 设置 ZMQ 服务器
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        host, port = address.split(':')
        self.socket.bind(f"tcp://0.0.0.0:{port}")
        self.socket.setsockopt(zmq.RCVHWM, 0)
        self.socket.setsockopt(zmq.SNDHWM, 0)
        
        self.model = None
        self.tokenizer = None
        self.current_requests = {}  # 跟踪当前处理的请求
        
        # 记录层分配
        self.my_layers = []  # 由这个worker负责的层
        
        logger.info(f"Worker {worker_id} started at {address}, listening on all interfaces")
    
    def store_kv_cache(self, request_id: str, layer_id: int, kv_tensor: torch.Tensor) -> bool:
        """将KV缓存存储到中央服务器"""
        if not self.kvcache_socket:
            return False
        
        # 将张量转换为numpy数组以便序列化
        kv_tensor_numpy = kv_tensor.cpu().numpy()
        
        # 发送存储请求
        request = {
            "command": "store",
            "request_id": request_id,
            "worker_id": self.worker_id,
            "layer_id": layer_id,
            "kv_tensor": kv_tensor_numpy,
            "data_size": kv_tensor_numpy.nbytes
        }
        
        self.kvcache_socket.send_pyobj(request)
        response = self.kvcache_socket.recv_pyobj()
        
        return response["status"] == "success"
    
    def retrieve_kv_cache(self, request_id: str, layer_id: int) -> Optional[torch.Tensor]:
        """从中央服务器检索KV缓存"""
        if not self.kvcache_socket:
            return None
        
        # 发送检索请求
        request = {
            "command": "retrieve",
            "request_id": request_id,
            "worker_id": self.worker_id,
            "layer_id": layer_id
        }
        
        self.kvcache_socket.send_pyobj(request)
        response = self.kvcache_socket.recv_pyobj()
        
        if response["status"] != "success":
            return None
        
        # 将numpy数组转换回张量
        kv_tensor = torch.from_numpy(response["kv_tensor"])
        if self.gpu_id is not None:
            # 放到对应的GPU上
            kv_tensor = kv_tensor.to(f"cuda:{self.gpu_id}")
        
        return kv_tensor
    
    def clear_kv_cache(self, request_id: str) -> bool:
        """清除KV缓存"""
        if not self.kvcache_socket:
            return False
        
        # 发送清除请求
        request = {
            "command": "clear",
            "request_id": request_id
        }
        
        self.kvcache_socket.send_pyobj(request)
        response = self.kvcache_socket.recv_pyobj()
        
        return response["status"] == "success"
    
    def load_model(self, model_path: str, gguf_path: str):
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # 获取模型配置
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        num_layers = config.num_hidden_layers
        
        # 计算该worker负责的层范围
        layers_per_worker = num_layers // self.num_workers
        remainder = num_layers % self.num_workers
        
        if self.worker_id < remainder:
            # 前remainder个worker多分配一层
            start_layer = self.worker_id * (layers_per_worker + 1)
            end_layer = start_layer + layers_per_worker + 1
        else:
            # 后面的worker正常分配
            start_layer = self.worker_id * layers_per_worker + remainder
            end_layer = start_layer + layers_per_worker
        
        self.my_layers = list(range(start_layer, end_layer))
        
        logger.info(f"Worker {self.worker_id} responsible for layers {self.my_layers}")
        
        # 设置内存分配
        max_memory = {
            0: "3GiB" if self.worker_id < 4 else None,    # 前4个进程使用 GPU 0，每个3GB
            1: "3GiB" if self.worker_id >= 4 else None,   # 后4个进程使用 GPU 1，每个3GB
            'cpu': "138GiB"  # CPU 内存
        }
        
        # 设置设备映射
        device_map = {
            'model.embed_tokens': 'cpu',
            'model.norm': 'cpu',
            'lm_head': 'cpu'
        }
        
        # 默认将所有层放在CPU上
        for i in range(num_layers):
            device_map[f'model.layers.{i}'] = 'cpu'
        
        logger.info(f"Loading model with device map strategy...")
        
        model_kwargs = {
            "device_map": device_map,
            "max_memory": max_memory,
            "torch_dtype": torch.float16,
            "low_cpu_mem_usage": True  # 添加这个选项来减少内存使用
        }
        
        try:
            # 使用自动设备映射
            logger.info(f"Loading model from {model_path} with specified device map...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                **model_kwargs
            )
            
            # 将不需要的层从内存中删除
            keep_layers = set(self.my_layers)
            for i in range(num_layers):
                if i not in keep_layers:
                    # 如果这一层不是由这个worker处理，释放其内存
                    if hasattr(self.model.model.layers, str(i)):
                        delattr(self.model.model.layers, str(i))
                    else:
                        self.model.model.layers[i] = None
                elif hasattr(self.model.model.layers, str(i)):
                    # 保留这个worker负责的层在CPU上
                    layer = getattr(self.model.model.layers, str(i))
                    # 当需要时移动到GPU
            
            # 打印设备分配情况
            for name, param in self.model.named_parameters():
                device = param.device
                if any(f'layers.{i}' in name for i in self.my_layers):
                    logger.info(f"  {name}: {device}")
            
            logger.info(f"Model loading complete. Worker {self.worker_id} is responsible for layers {self.my_layers}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
        
        self.model.eval()
        
        # 拦截前向传播方法进行分布式计算
        original_forward = self.model.forward
        
        def distributed_forward(*args, **kwargs):
            """支持分布式推理的前向传播"""
            with torch.no_grad():
                # 生成唯一的请求ID
                request_id = str(uuid.uuid4())
                self.current_requests[request_id] = True
                
                try:
                    logger.info(f"Starting distributed forward pass for request {request_id}")
                    
                    # 将我负责的层移到GPU上
                    for layer_idx in self.my_layers:
                        if hasattr(self.model.model.layers, str(layer_idx)):
                            layer = getattr(self.model.model.layers, str(layer_idx))
                            self.memory_manager.move_to_gpu(layer)
                        else:
                            layer = self.model.model.layers[layer_idx]
                            self.memory_manager.move_to_gpu(layer)
                    
                    # 正常调用原始forward方法
                    result = original_forward(*args, **kwargs)
                    
                    # 完成后将层移回CPU
                    for layer_idx in self.my_layers:
                        if hasattr(self.model.model.layers, str(layer_idx)):
                            layer = getattr(self.model.model.layers, str(layer_idx))
                            self.memory_manager.move_to_cpu(layer)
                        else:
                            layer = self.model.model.layers[layer_idx]
                            self.memory_manager.move_to_cpu(layer)
                    
                    # 清理
                    torch.cuda.empty_cache()
                    
                    return result
                    
                finally:
                    # 清理请求记录
                    if request_id in self.current_requests:
                        del self.current_requests[request_id]
        
        # 替换模型的forward方法
        self.model.forward = distributed_forward
    
    class CustomStreamer(TextStreamer):
        """自定义文本生成流式处理器"""
        
        def __init__(self, tokenizer, on_text_chunk, **kwargs):
            super().__init__(tokenizer, **kwargs)
            self.on_text_chunk = on_text_chunk
            self.generated_text = ""
        
        def on_finalized_text(self, text: str, stream_end: bool = False):
            # 这里不进行标准输出
            if text:
                self.generated_text += text
                self.on_text_chunk(text, stream_end)
    
    def generate(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # 解析输入参数
            input_ids = torch.tensor(input_data['input_ids'])
            attention_mask = torch.tensor(input_data['attention_mask'])
            max_new_tokens = input_data.get('max_new_tokens', 512)
            temperature = input_data.get('temperature', 0.7)
            top_p = input_data.get('top_p', 0.9)
            
            # 生成唯一的请求ID
            request_id = str(uuid.uuid4())
            self.current_requests[request_id] = True
            
            # 移动输入到GPU
            gpu_id = self.gpu_id
            if gpu_id is not None:
                input_ids = input_ids.to(f"cuda:{gpu_id}")
                attention_mask = attention_mask.to(f"cuda:{gpu_id}")
            
            # 流式生成处理器
            class StreamingHandler:
                def __init__(self, socket, tokenizer):
                    self.socket = socket
                    self.tokenizer = tokenizer
                    self.current_text = ""
                    self.last_send_time = time.time()
                    
                def __call__(self, text_chunk: str, stream_end: bool = False):
                    current_time = time.time()
                    self.current_text += text_chunk
                    
                    # 每0.5秒或结束时发送一次
                    if stream_end or (current_time - self.last_send_time) > 0.5:
                        self.socket.send_pyobj({
                            'status': 'streaming',
                            'text': self.current_text,
                            'finished': stream_end
                        })
                        # 等待客户端确认
                        self.socket.recv_pyobj()
                        self.last_send_time = current_time
            
            # 创建流式处理器
            streamer = self.CustomStreamer(
                self.tokenizer,
                on_text_chunk=StreamingHandler(self.socket, self.tokenizer),
                skip_prompt=True,
                skip_special_tokens=True
            )
            
            # 拦截原始forward方法进行KV缓存共享
            original_forward = self.model.forward
            
            def shared_kv_forward(*args, **kwargs):
                """使用共享KV缓存的前向传播"""
                with torch.no_grad():
                    # 默认使用原始forward
                    result = original_forward(*args, **kwargs)
                    
                    # 获取和存储KV缓存
                    if hasattr(result, 'past_key_values') and result.past_key_values:
                        past_key_values = result.past_key_values
                        
                        # 存储我负责的层的KV缓存
                        for layer_idx in self.my_layers:
                            if layer_idx < len(past_key_values):
                                kv_cache = past_key_values[layer_idx]
                                # 存储KV缓存
                                self.store_kv_cache(request_id, layer_idx, kv_cache)
                        
                        # 替换其他层的KV缓存为从服务器获取的值
                        new_past_key_values = list(past_key_values)
                        for layer_idx in range(len(past_key_values)):
                            if layer_idx not in self.my_layers:
                                # 从服务器获取这一层的KV缓存
                                remote_kv_cache = self.retrieve_kv_cache(request_id, layer_idx)
                                if remote_kv_cache is not None:
                                    new_past_key_values[layer_idx] = remote_kv_cache
                        
                        # 使用更新后的KV缓存
                        result.past_key_values = tuple(new_past_key_values)
                    
                    return result
            
            # 如果有KV缓存服务器，使用共享KV缓存forward
            if self.kvcache_socket:
                self.model.forward = shared_kv_forward
            
            # 生成文本
            logger.info("Starting text generation...")
            start_time = time.time()
            with torch.no_grad(), torch.amp.autocast('cuda'):
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=input_data.get('do_sample', True),
                    streamer=streamer
                )
            
            # 恢复原始forward方法
            if self.kvcache_socket:
                self.model.forward = original_forward
                # 清理KV缓存
                self.clear_kv_cache(request_id)
            
            # 记录性能统计
            end_time = time.time()
            generation_time = end_time - start_time
            tokens_generated = len(outputs[0]) - len(input_ids[0])
            
            logger.info("Generation statistics:")
            logger.info(f"- Total time: {generation_time:.2f} seconds")
            logger.info(f"- Tokens generated: {tokens_generated}")
            logger.info(f"- Speed: {tokens_generated/generation_time:.2f} tokens/sec")
            
            # 清理当前请求记录
            if request_id in self.current_requests:
                del self.current_requests[request_id]
            
            # 最终返回完整的 token ids
            return {
                'status': 'success',
                'output_ids': outputs.cpu().numpy().tolist()
            }
        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }

    def run(self):
        logger.info("Worker ready to receive requests")
        try:
            while True:
                try:
                    message = self.socket.recv_pyobj()
                    command = message.get('command')
                    
                    if command == 'load_model':
                        self.load_model(message['model_path'], message['gguf_path'])
                        self.socket.send_pyobj({'status': 'success'})
                    
                    elif command == 'generate':
                        result = self.generate(message['data'])
                        self.socket.send_pyobj(result)
                    
                    elif command == 'shutdown':
                        logger.info("Received shutdown command")
                        self.socket.send_pyobj({'status': 'shutdown'})
                        # 清理资源
                        if hasattr(self, 'model'):
                            del self.model
                        if hasattr(self, 'tokenizer'):
                            del self.tokenizer
                        torch.cuda.empty_cache()
                        logger.info("Resources cleaned up")
                        break
                    
                except Exception as e:
                    logger.error(f"Error processing message: {str(e)}")
                    self.socket.send_pyobj({'status': 'error', 'error': str(e)})
        finally:
            # 确保资源被清理
            logger.info("Cleaning up worker resources...")
            self.socket.close()
            self.context.term()
            if hasattr(self, 'kvcache_socket') and self.kvcache_socket:
                self.kvcache_socket.close()
                self.kvcache_context.term()
            logger.info("Worker shutdown complete")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_port', type=int, default=29500)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--worker_address', type=str, required=False)
    parser.add_argument('--worker_id', type=int, default=0)
    parser.add_argument('--kvcache_address', type=str, default=None,
                        help='Address of KV Cache Server (e.g., "10.21.22.206:29600")')
    args = parser.parse_args()
    
    if args.worker_address:
        # 单个 worker 进程模式
        worker = ModelWorker(args.worker_address, args.worker_id, args.num_workers, args.kvcache_address)
        worker.run()
    else:
        # 主进程模式，启动多个 worker
        for worker_id in range(args.num_workers):
            start_worker_process(worker_id, args.base_port, args.num_workers, args.kvcache_address)
        
        # 等待所有进程
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down all workers...")

def start_worker_process(worker_id: int, base_port: int, num_workers: int, kvcache_address: str = None):
    """启动单个 worker 进程"""
    # 计算 NUMA 节点
    numa_nodes = get_numa_nodes()
    node_id = 0 if worker_id < 4 else 1  # 前4个进程在节点0，后4个在节点1
    
    # 计算端口
    port = base_port + worker_id
    
    # 使用 numactl 启动进程
    cmd = [
        'numactl',
        f'--cpunodebind={node_id}',
        f'--membind={node_id}',
        'python',
        '-m',
        'ktransformers.worker',
        f'--worker_address=10.21.22.206:{port}',
        f'--worker_id={worker_id}',
        f'--num_workers={num_workers}'
    ]
    
    # 添加KV缓存服务器地址（如果有）
    if kvcache_address:
        cmd.append(f'--kvcache_address={kvcache_address}')
    
    logger.info(f"Starting worker {worker_id} on NUMA node {node_id} with port {port}")
    subprocess.Popen(cmd)

if __name__ == '__main__':
    main() 