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

class KVCacheLayer(torch.nn.Module):
    """代理层，用于从KV Cache获取结果"""
    
    def __init__(self, worker, layer_id):
        super().__init__()
        self.worker = worker
        self.layer_id = layer_id
    
    def forward(self, hidden_states, attention_mask=None, **kwargs):
        """从KV缓存获取结果，而不是本地计算"""
        request_id = kwargs.get('request_id', str(uuid.uuid4()))
        
        # 从KV缓存服务器获取计算结果
        result = self.worker.retrieve_kv_cache(request_id, self.layer_id)
        
        if result is None:
            # 如果结果不可用，记录错误
            logger.error(f"Layer {self.layer_id} result not available in KV cache")
            # 创建一个形状匹配的张量，但全是0
            # 这不是理想的解决方案，但可以防止程序崩溃
            return torch.zeros_like(hidden_states)
        
        return result

class ModelWorker:
    def __init__(self, address: str, worker_id: int = 0, num_workers: int = 1, kvcache_address: str = None, embedding_address: str = None):
        self.worker_id = worker_id
        self.num_workers = num_workers
        self.kvcache_address = kvcache_address
        self.embedding_address = embedding_address
        
        # 如果提供了KV Cache服务器地址，则连接
        if self.kvcache_address:
            self.kvcache_context = zmq.Context()
            self.kvcache_socket = self.kvcache_context.socket(zmq.REQ)
            self.kvcache_socket.connect(f"tcp://{kvcache_address}")
            logger.info(f"Worker {worker_id} connected to KV Cache Server at {kvcache_address}")
        else:
            self.kvcache_socket = None
            logger.info(f"Worker {worker_id} running without KV Cache Server")
        
        # 如果提供了嵌入服务器地址，则连接
        if self.embedding_address:
            self.embedding_context = zmq.Context()
            self.embedding_socket = self.embedding_context.socket(zmq.REQ)
            self.embedding_socket.connect(f"tcp://{embedding_address}")
            logger.info(f"Worker {worker_id} connected to Embedding Server at {embedding_address}")
        else:
            self.embedding_socket = None
            logger.info(f"Worker {worker_id} running without Embedding Server")
        
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
        """加载模型并分配层"""
        logger.info(f"Worker {self.worker_id} loading model from {model_path}")
        
        # 加载配置
        logger.info(f"Loading model config...")
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        
        # 获取模型总层数
        if hasattr(config, 'num_hidden_layers'):
            total_layers = config.num_hidden_layers
        elif hasattr(config, 'n_layers'):
            total_layers = config.n_layers
        else:
            total_layers = 32  # 默认值
        
        logger.info(f"Model has {total_layers} layers in total")
        
        # 计算每个worker负责的层
        layers_per_worker = total_layers // self.num_workers
        start_layer = self.worker_id * layers_per_worker
        end_layer = start_layer + layers_per_worker if self.worker_id < self.num_workers - 1 else total_layers
        
        self.my_layers = list(range(start_layer, end_layer))
        logger.info(f"Worker {self.worker_id} is responsible for layers {self.my_layers}")
        
        # 设置设备映射 - 关键改进
        # 我们将只加载自己负责的层到内存中
        device_map = {}
        
        # 将基础模型放在CPU上
        device_map['model.embed_tokens'] = 'cpu'
        device_map['model.norm'] = 'cpu'
        device_map['lm_head'] = 'cpu'
        
        # 只将自己负责的层映射到此worker
        for i in range(total_layers):
            if i in self.my_layers:
                # 我们负责的层 - 初始放在CPU上，计算时会移到GPU
                device_map[f'model.layers.{i}'] = 'cpu'
            else:
                # 不负责的层映射到'meta'设备(不加载参数)
                device_map[f'model.layers.{i}'] = 'meta'
        
        logger.info(f"Device map for worker {self.worker_id}: {device_map}")
        
        # 加载模型 - 只加载我们负责的层
        logger.info(f"Loading model (only responsible layers)...")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                config=config,
                device_map=device_map,  # 使用我们自定义的设备映射
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            
            # 替换不负责的层使用KV缓存层
            model_layers = self.model.model.layers
            for i in range(total_layers):
                if i not in self.my_layers and hasattr(model_layers, str(i)):
                    logger.info(f"Replacing layer {i} with KVCacheLayer proxy")
                    # 创建一个代理层，它会从KV缓存服务器获取结果
                    model_layers[i] = KVCacheLayer(self, i)
            
            logger.info(f"Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
        
        # 加载tokenizer
        logger.info(f"Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        logger.info(f"Tokenizer loaded successfully")
        
        # 设置模型的钩子，管理GPU内存
        self.setup_model_hooks()
    
    def setup_model_hooks(self):
        """设置模型钩子，以便在推理过程中处理层交换"""
        # 获取所有层
        if not hasattr(self.model, 'model') or not hasattr(self.model.model, 'layers'):
            logger.warning("无法获取模型层，跳过钩子设置")
            return
        
        layers = self.model.model.layers
        
        # 只处理我们负责的层
        for layer_id in self.my_layers:
            if not hasattr(layers, str(layer_id)):
                continue
            
            layer = layers[layer_id]
            
            # 保存原始forward方法
            if not hasattr(layer, '_original_forward'):
                layer._original_forward = layer.forward
            
            # 创建新的forward方法
            def make_hook(*args, **kwargs):
                # 移动到GPU
                logger.debug(f"Moving layer {layer_id} to GPU")
                self.memory_manager.move_to_gpu(layers[layer_id])
                
                # 执行计算
                result = layer._original_forward(*args, **kwargs)
                
                # 获取request_id并存储结果到KV缓存
                request_id = kwargs.get('request_id', None)
                if request_id and self.kvcache_socket:
                    self.store_kv_cache(request_id, layer_id, result)
                
                # 计算后移回CPU
                logger.debug(f"Moving layer {layer_id} back to CPU")
                self.memory_manager.move_to_cpu(layers[layer_id])
                
                return result
            layer.forward = make_hook
        
        logger.info(f"Set up hooks for {len(self.my_layers)} layers")
    
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
    
    def create_streamer(self, request_id):
        """创建一个流式输出器，用于在生成过程中返回部分结果"""
        self.socket.send_pyobj({
            'status': 'streaming',
            'text': '',
            'token_ids': []
        })
        
        # 等待客户端确认
        ack = self.socket.recv_pyobj()
        if ack.get('status') != 'continue':
            logger.warning(f"Client did not send continue status, got: {ack}")
        
        class ZMQStreamer:
            def __init__(self, worker, socket, request_id):
                self.worker = worker
                self.socket = socket
                self.request_id = request_id
                self.generated_text = ""
                self.token_ids = []
            
            def put(self, token_ids):
                """接收新生成的token，发送到客户端"""
                # 如果是单个token，包装成列表
                if not isinstance(token_ids, list):
                    token_ids = [token_ids]
                
                # 添加新token到累积列表
                self.token_ids.extend(token_ids)
                
                # 解码文本
                new_text = self.worker.tokenizer.decode(
                    self.token_ids, 
                    skip_special_tokens=True
                )
                # 只发送新增部分
                delta_text = new_text[len(self.generated_text):]
                self.generated_text = new_text
                
                # 发送更新
                self.socket.send_pyobj({
                    'status': 'streaming',
                    'text': delta_text,
                    'token_ids': token_ids
                })
                
                # 等待客户端确认
                ack = self.socket.recv_pyobj()
                if ack.get('status') != 'continue':
                    logger.warning(f"Client did not send continue status, got: {ack}")
            
            def end(self):
                """标记生成完成"""
                logger.info("Streaming completed")
        
        return ZMQStreamer(self, self.socket, request_id)
    
    def generate(self, input_data):
        """使用分布式层处理生成tokens"""
        try:
            # 提取输入数据
            input_ids = torch.tensor(input_data.get('input_ids'))
            attention_mask = torch.tensor(input_data.get('attention_mask'))
            max_new_tokens = input_data.get('max_new_tokens', 50)
            temperature = input_data.get('temperature', 0.7)
            top_p = input_data.get('top_p', 0.9)
            
            # 如果有embedding服务器，获取输入嵌入
            if hasattr(self, 'embedding_socket') and self.embedding_socket:
                logger.info("Using embedding server for input embeddings")
                embedding_request = {
                    "command": "embed_tokens",
                    "input_ids": input_ids.numpy().tolist()
                }
                self.embedding_socket.send_pyobj(embedding_request)
                embedding_response = self.embedding_socket.recv_pyobj()
                
                if embedding_response.get('status') == 'success':
                    # 使用从embedding服务器获取的嵌入
                    embeds = torch.tensor(embedding_response.get('embeddings')).to(self.device)
                    logger.info(f"Got embeddings from server, shape: {embeds.shape}")
                else:
                    logger.error("Failed to get embeddings from server")
            
            # 创建一个唯一的请求ID
            request_id = str(uuid.uuid4())
            self.current_requests[request_id] = {'start_time': time.time()}
            
            # 记录输入数据大小
            logger.info(f"Input shape: {input_ids.shape}")
            
            # 创建流式输出器
            streamer = None
            if input_data.get('stream', True):
                streamer = self.create_streamer(request_id)
            
            # 保存原始forward方法
            original_forward = self.model.forward
            
            # 修改模型的forward方法以使用request_id
            def forward_with_request_id(*args, **kwargs):
                kwargs['request_id'] = request_id
                return self.model.forward(*args, **kwargs)
            
            # 应用修改后的forward方法
            self.model.forward = forward_with_request_id
            
            # 记录开始时间
            start_time = time.time()
            
            # 生成输出
            logger.info(f"Starting generation with max_new_tokens={max_new_tokens}")
            outputs = self.model.generate(
                input_ids=input_ids.unsqueeze(0) if input_ids.dim() == 1 else input_ids,
                attention_mask=attention_mask.unsqueeze(0) if attention_mask.dim() == 1 else attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=input_data.get('do_sample', True),
                streamer=streamer
            )
            
            # 恢复原始forward方法
            self.model.forward = original_forward
            
            # 清理KV缓存
            if self.kvcache_socket:
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
    parser.add_argument('--embedding_address', type=str, default=None,
                        help='Address of Embedding Server (e.g., "10.21.22.206:29700")')
    args = parser.parse_args()
    
    if args.worker_address:
        # 单个 worker 进程模式
        worker = ModelWorker(args.worker_address, args.worker_id, args.num_workers, args.kvcache_address, args.embedding_address)
        worker.run()
    else:
        # 主进程模式，启动多个 worker
        for worker_id in range(args.num_workers):
            start_worker_process(worker_id, args.base_port, args.num_workers, args.kvcache_address, args.embedding_address)
        
        # 等待所有进程
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down all workers...")

def start_worker_process(worker_id: int, base_port: int, num_workers: int, kvcache_address: str = None, embedding_address: str = None):
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
    
    # 添加嵌入服务器地址（如果有）
    if embedding_address:
        cmd.append(f'--embedding_address={embedding_address}')
    
    logger.info(f"Starting worker {worker_id} on NUMA node {node_id} with port {port}")
    subprocess.Popen(cmd)

if __name__ == '__main__':
    main() 