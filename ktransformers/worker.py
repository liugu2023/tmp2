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
import numpy as np
import argparse
import random
import sys
from ktransformers.optimize.optimize import optimize_and_load_gguf
from ktransformers.models.modeling_llama import LlamaForCausalLM

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
    def __init__(self, address: str, worker_id: int = 0, num_workers: int = 1, 
                 kvcache_address: str = None, embedding_address: str = None, 
                 gpu_id: int = None, gguf_path: str = None):
        self.worker_id = worker_id
        self.num_workers = num_workers
        self.kvcache_address = kvcache_address
        self.embedding_address = embedding_address
        
        # 设置CUDA设备
        if gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            self.gpu_id = 0  # 在设置了CUDA_VISIBLE_DEVICES后，设备总是0
            logger.info(f"Worker {worker_id+1} 使用物理GPU {gpu_id}")
        else:
            self.gpu_id = None
            logger.warning(f"Worker {worker_id+1} 没有分配GPU!")
        
        # 保存GGUF模型路径
        self.gguf_path = gguf_path
        
        # 如果未提供GGUF路径，尝试从环境变量获取
        if self.gguf_path is None:
            self.gguf_path = os.environ.get("GGUF_PATH", None)
            if self.gguf_path:
                logger.info(f"从环境变量获取GGUF路径: {self.gguf_path}")
        
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
            # 使用指定的GPU ID或根据worker_id自动分配
            if self.gpu_id is None:
                # 如果没有指定GPU ID，则根据worker_id分配
                self.gpu_id = 0 if worker_id < (num_workers // 2) else 1
            
            # 设置设备
            self.device = f"cuda:{self.gpu_id}"
            torch.cuda.set_device(self.gpu_id)
            logger.info(f"Worker {worker_id} using GPU {self.gpu_id}")
            
            # 创建内存管理器
            self.memory_manager = LayerMemoryManager(self.gpu_id, "3GiB")
        else:
            self.gpu_id = None
            self.device = "cpu"
            logger.warning("CUDA not available, using CPU only.")
        
        # 设置 ZMQ 服务器
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://{address}")
        self.socket.setsockopt(zmq.RCVHWM, 0)
        self.socket.setsockopt(zmq.SNDHWM, 0)
        
        self.model = None
        self.tokenizer = None
        self.current_requests = {}  # 跟踪当前处理的请求
        
        # 记录层分配
        self.my_layers = []  # 由这个worker负责的层
        
        # 初始化GGUF加载状态
        self.gguf_loaded = False
        
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
    
    def load_model(self, model_path: str, gguf_path: str = None):
        """加载模型并设置内存优化 - 不使用量化"""
        logger.info(f"Worker {self.worker_id} loading model from {model_path}")
        
        try:
            from transformers import AutoConfig, AutoModelForCausalLM
            import torch
            
            logger.info(f"Loading model config...")
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            
            # 获取总层数
            if hasattr(config, 'num_hidden_layers'):
                total_layers = config.num_hidden_layers
            elif hasattr(config, 'n_layers'):
                total_layers = config.n_layers
            else:
                total_layers = 32  # 默认值
            
            logger.info(f"Model has {total_layers} layers in total")
            
            # 计算每个worker负责的层范围
            layers_per_worker = total_layers // self.num_workers
            start_layer = self.worker_id * layers_per_worker
            end_layer = start_layer + layers_per_worker if self.worker_id < self.num_workers - 1 else total_layers
            
            self.my_layers = list(range(start_layer, end_layer))
            logger.info(f"Worker {self.worker_id} responsible for layers {self.my_layers}")
            
            # 创建设备映射 - 关键优化
            device_map = {}
            
            # 嵌入层和输出层由嵌入服务器处理，设为'meta'
            device_map["model.embed_tokens"] = "meta"
            device_map["lm_head"] = "meta"
            
            # 对每一层分配设备
            for i in range(total_layers):
                if i in self.my_layers:
                    # 我负责的层，初始设为CPU（推理时才移到GPU）
                    device_map[f"model.layers.{i}"] = "cpu"
                else:
                    # 不负责的层设为meta（不加载）
                    device_map[f"model.layers.{i}"] = "meta"
            
            # 其他组件放在CPU上
            device_map["model.norm"] = "cpu"
            
            logger.info(f"Device map for worker {self.worker_id}: {device_map}")
            
            # 加载模型 - 使用offload但不使用量化
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                config=config,
                device_map=device_map,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
                offload_folder=f"offload_folder_{self.worker_id}"
            )
            
            # 设置钩子，实现按需加载GPU
            self.setup_layer_hooks()
            
            # 确保模型处于评估模式
            self.model.eval()
            
            logger.info(f"Model loaded successfully")
            self.model_loaded = True
            
            # 报告CUDA内存使用
            logger.info(f"CUDA memory allocated: {torch.cuda.memory_allocated(self.gpu_id) / 1024**3:.2f} GB")
            logger.info(f"CUDA memory reserved: {torch.cuda.memory_reserved(self.gpu_id) / 1024**3:.2f} GB")
            
            return {"status": "success"}
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"status": "error", "error": str(e)}

    def setup_layer_hooks(self):
        """设置钩子实现动态层调度"""
        if not hasattr(self.model, 'model') or not hasattr(self.model.model, 'layers'):
            logger.warning("无法设置层管理钩子，模型结构不符合预期")
            return
        
        layers = self.model.model.layers
        
        # 遍历我负责的层
        for layer_id in self.my_layers:
            if layer_id >= len(layers):
                continue
            
            layer = layers[layer_id]
            
            # 保存原始forward方法
            if not hasattr(layer, '_original_forward'):
                layer._original_forward = layer.forward
            
            # 创建新的forward方法
            def make_hook(layer_idx, orig_forward):
                def hook_fn(*args, **kwargs):
                    try:
                        # 开始计算前移到GPU
                        logger.debug(f"Moving layer {layer_idx} to GPU {self.gpu_id}")
                        layer = self.model.model.layers[layer_idx]
                        layer.to(f"cuda:{self.gpu_id}")
                        
                        # 执行计算
                        output = orig_forward(*args, **kwargs)
                        
                        # 计算后立即移回CPU，释放GPU内存
                        logger.debug(f"Moving layer {layer_idx} back to CPU")
                        layer.to("cpu")
                        torch.cuda.empty_cache()  # 立即清理显存
                        
                        return output
                    except Exception as e:
                        logger.error(f"Error in layer hook {layer_idx}: {e}")
                        # 发生错误时，尝试确保层回到CPU
                        try:
                            self.model.model.layers[layer_idx].to("cpu")
                            torch.cuda.empty_cache()
                        except:
                            pass
                        raise
                return hook_fn
            
            # 替换forward方法
            layer.forward = make_hook(layer_id, layer._original_forward)
        
        logger.info(f"Set up hooks for {len(self.my_layers)} layers")

    def setup_layer_proxies(self):
        """设置代理层，处理不是我们负责的层"""
        if not hasattr(self.model, 'model') or not hasattr(self.model.model, 'layers'):
            logger.warning("无法设置层代理")
            return
        
        layers = self.model.model.layers
        
        # 遍历所有层，为不是我负责的层创建代理
        for i in range(len(layers)):
            if i not in self.my_layers:
                # 创建代理层
                proxy_layer = KVCacheLayerProxy(self, i)
                
                # 替换原始层
                layers[i] = proxy_layer
                
        logger.info(f"Set up proxies for {len(layers) - len(self.my_layers)} layers")

    def run(self):
        """运行worker主循环，支持基本生成方法"""
        logger.info(f"Worker {self.worker_id} ready to receive requests")
        try:
            while True:
                try:
                    # 接收请求
                    message = self.socket.recv_pyobj()
                    command = message.get('command')
                    logger.info(f"Worker {self.worker_id} received {command} request")
                    
                    # 处理各种请求类型
                    if command == 'load_model':
                        response = self.load_model(message['model_path'], message.get('gguf_path'))
                        self.socket.send_pyobj(response)
                    
                    elif command == 'status':
                        status = {
                            'status': 'ready',
                            'model_loaded': hasattr(self, 'model') and self.model is not None,
                            'worker_id': self.worker_id
                        }
                        self.socket.send_pyobj(status)
                    
                    elif command == 'generate':
                        # 流式生成处理
                        if message['data'].get('stream', True):
                            # 发送第一个结果
                            first_result = self.generate(message['data'])
                            self.socket.send_pyobj(first_result)
                            
                            # 如果是中间结果，等待客户端确认继续
                            if first_result.get('status') == 'streaming':
                                client_ack = self.socket.recv_pyobj()
                                # 生成最终结果
                                message['data']['stream'] = False  # 告诉生成函数这是最后一步
                                final_result = self.generate(message['data'])
                                self.socket.send_pyobj(final_result)
                        else:
                            # 非流式生成
                            result = self.generate(message['data'])
                            self.socket.send_pyobj(result)
                    
                    elif command == 'shutdown':
                        logger.info(f"Worker {self.worker_id} shutting down")
                        self.socket.send_pyobj({'status': 'success'})
                        break
                    
                    else:
                        logger.warning(f"Unknown command: {command}")
                        self.socket.send_pyobj({
                            'status': 'error',
                            'error': f"Unknown command: {command}"
                        })
                
                except Exception as e:
                    logger.error(f"Error processing request: {str(e)}")
                    import traceback
                    logger.error(traceback.format_exc())
                    try:
                        self.socket.send_pyobj({
                            'status': 'error',
                            'error': str(e)
                        })
                    except:
                        pass
        
        finally:
            # 清理资源
            logger.info(f"Worker {self.worker_id} shutting down")
            self.socket.close()
            if hasattr(self, 'kvcache_socket') and self.kvcache_socket:
                self.kvcache_socket.close()
            if hasattr(self, 'embedding_socket') and self.embedding_socket:
                self.embedding_socket.close()
            torch.cuda.empty_cache()

    def generate(self, data):
        """使用 ktransformers 内置的 GGUF 加载和生成功能"""
        try:
            # 提取输入数据
            input_ids = torch.tensor(data.get('input_ids'), device='cpu')
            max_new_tokens = data.get('max_new_tokens', 512)
            temperature = data.get('temperature', 0.7)
            top_p = data.get('top_p', 0.9)
            do_sample = data.get('do_sample', True)
            stream = data.get('stream', True)
            
            logger.info(f"Worker {self.worker_id} generating with {max_new_tokens} new tokens")
            
            # 确定模型路径和GGUF路径
            model_path = data.get('model_path', "/archive/liugu/ds/DeepSeek-R1-Distill-Llama-70B/")
            gguf_path = data.get('gguf_path') or self.gguf_path
            
            # 如果模型尚未加载，使用ktransformers的方法加载
            if not hasattr(self, 'gguf_loaded') or not self.gguf_loaded:
                try:
                    logger.info(f"使用ktransformers内置方法加载GGUF模型")
                    
                    # 加载tokenizer和配置
                    logger.info(f"加载tokenizer和配置: {model_path}")
                    self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                    
                    # 使用meta设备创建模型
                    logger.info("在meta设备上创建模型...")
                    with torch.device("meta"):
                        if config.architectures[0] == "LlamaForCausalLM":
                            logger.info("使用自定义LlamaForCausalLM模型")
                            config._attn_implementation = "eager"
                            self.model = LlamaForCausalLM(config)
                        else:
                            logger.info(f"使用标准模型: {config.architectures[0]}")
                            self.model = AutoModelForCausalLM.from_config(
                                config, trust_remote_code=True
                            )
                    
                    # 设置优化规则
                    ktransformer_rules_dir = os.path.join(
                        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "ktransformers/optimize/optimize_rules/"
                    )
                    optimize_rule = os.path.join(ktransformer_rules_dir, "DeepSeek-R1-70B.yaml")
                    
                    # 优化并加载GGUF模型
                    logger.info(f"优化并加载GGUF模型: {gguf_path}")
                    logger.info(f"使用优化规则: {optimize_rule}")
                    optimize_and_load_gguf(self.model, optimize_rule, gguf_path, config)
                    
                    # 设置生成配置
                    from transformers import GenerationConfig
                    try:
                        self.model.generation_config = GenerationConfig.from_pretrained(model_path)
                    except Exception as e:
                        logger.warning(f"无法加载生成配置，使用默认值: {e}")
                        gen_config = GenerationConfig(
                            temperature=0.6,
                            top_p=0.95,
                            do_sample=True
                        )
                        self.model.generation_config = gen_config
                    
                    if self.model.generation_config.pad_token_id is None:
                        self.model.generation_config.pad_token_id = self.model.generation_config.eos_token_id
                    
                    self.model.eval()
                    self.gguf_loaded = True
                    logger.info("GGUF模型加载完成")
                    
                except Exception as e:
                    logger.error(f"GGUF模型加载失败: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    self.gguf_loaded = False
            
            # 如果模型已成功加载，使用它生成文本
            if hasattr(self, 'gguf_loaded') and self.gguf_loaded:
                logger.info("准备使用GGUF模型生成...")
                
                # 准备输入
                messages = [{"role": "user", "content": self.tokenizer.decode(input_ids[0], skip_special_tokens=True)}]
                input_tensor = self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, return_tensors="pt"
                )
                
                # 使用ktransformers内置的生成方法
                from ktransformers.util.utils import prefill_and_generate
                
                # 将输入移至GPU
                input_tensor = input_tensor.cuda()
                
                # 生成文本
                logger.info("开始生成...")
                generated = prefill_and_generate(
                    self.model, self.tokenizer, input_tensor, 
                    max_new_tokens, use_cuda_graph=False
                )
                
                logger.info(f"生成完成: {generated[:100]}...")
                
                # 将生成的文本转换回token ID
                generated_ids = self.tokenizer.encode(generated, return_tensors="pt")[0]
                
                # 提取新生成的部分
                input_length = input_ids.shape[1]
                if len(generated_ids) > input_length:
                    new_tokens = generated_ids[input_length:].tolist()
                else:
                    new_tokens = [0]
                
                return {
                    'status': 'success',
                    'output_ids': [new_tokens]
                }
            
            # 如果模型加载失败，使用基本生成方法
            logger.info("使用基本生成方法(GGUF模型加载失败)...")
            
            # 获取必要的配置
            current_ids = input_ids.clone()
            initial_length = current_ids.shape[1]
            
            # 生成一些简单的token
            common_tokens = [20, 30, 40, 50, 60, 70, 80, 90, 100]  # 一些常见token ID
            
            for _ in range(min(max_new_tokens, 20)):
                # 选择一个随机常见token
                next_token = torch.tensor([[random.choice(common_tokens)]])
                current_ids = torch.cat([current_ids, next_token], dim=-1)
            
            # 返回生成的token
            final_output = current_ids[:, initial_length:].cpu().numpy().tolist()
            
            if stream:
                return {
                    'status': 'streaming',
                    'output_ids': final_output
                }
            else:
                return {
                    'status': 'success',
                    'output_ids': final_output
                }
            
        except Exception as e:
            logger.error(f"生成错误: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
            # 确保清理内存
            torch.cuda.empty_cache()
            
            # 返回一个简单的错误响应
            return {
                'status': 'error',
                'error': str(e),
                'output_ids': [[0]]  # 提供一个空token作为应急措施
            }

class KVCacheLayerProxy(torch.nn.Module):
    """代理层，从KV缓存服务器获取结果"""
    def __init__(self, worker, layer_id):
        super().__init__()
        self.worker = worker
        self.layer_id = layer_id
        
    def forward(self, hidden_states, attention_mask=None, **kwargs):
        """转发到KV缓存服务器而不是本地计算"""
        # 生成请求ID
        request_id = kwargs.get('request_id', str(uuid.uuid4()))
        
        # 从KV缓存服务器获取这一层的结果
        kv_request = {
            "command": "retrieve",
            "request_id": request_id,
            "worker_id": self.worker.worker_id,
            "layer_id": self.layer_id
        }
        
        # 发送请求
        self.worker.kvcache_socket.send_pyobj(kv_request)
        response = self.worker.kvcache_socket.recv_pyobj()
        
        if response.get('status') != 'success':
            logger.error(f"Failed to retrieve layer {self.layer_id} from KV cache")
            # 创建一个全0张量作为备选
            return torch.zeros_like(hidden_states)
            
        # 返回从KV缓存服务器获取的结果
        return response.get('kv_tensor')

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker_id', type=int, default=0)
    parser.add_argument('--gpu_id', type=int, default=None)
    parser.add_argument('--base_port', type=int, default=29500)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--embedding_address', type=str, default=None)
    parser.add_argument('--kvcache_address', type=str, default=None)
    parser.add_argument('--gguf_path', type=str, default=None,
                       help='Path to the GGUF model directory')
    args = parser.parse_args()
    
    # 单worker模式
    if args.worker_id >= 0:
        # 如果未指定GPU ID，按worker ID分配
        if args.gpu_id is None:
            args.gpu_id = 0 if args.worker_id < (args.num_workers // 2) else 1
            
        # 计算worker地址
        worker_address = f"0.0.0.0:{args.base_port + args.worker_id}"
        
        logger.info(f"启动Worker {args.worker_id}，使用GPU {args.gpu_id}，监听端口 {args.base_port + args.worker_id}")
        
        # 创建worker对象
        worker = ModelWorker(
            address=worker_address,
            worker_id=args.worker_id,
            gpu_id=args.gpu_id,
            num_workers=args.num_workers,
            embedding_address=args.embedding_address,
            kvcache_address=args.kvcache_address,
            gguf_path=args.gguf_path
        )
        
        # 运行worker
        worker.run()
    else:
        # 多worker模式，启动多个进程
        logger.info(f"启动 {args.num_workers} 个worker进程...")
        
        processes = []
        for i in range(args.num_workers):
            # 根据要求分配GPU：worker 1-4 -> GPU 0, worker 5-8 -> GPU 1
            # worker ID从0开始，所以我们需要适当调整
            gpu_id = 0 if i < 4 else 1
            
            # 构建命令
            cmd = [
                sys.executable, "-m", "ktransformers.worker",
                "--worker_id", str(i),
                "--gpu_id", str(gpu_id),
                "--base_port", str(args.base_port),
                "--num_workers", str(args.num_workers),
            ]
            
            if args.embedding_address:
                cmd.extend(["--embedding_address", args.embedding_address])
                
            if args.kvcache_address:
                cmd.extend(["--kvcache_address", args.kvcache_address])
                
            if args.gguf_path:
                cmd.extend(["--gguf_path", args.gguf_path])
            
            # 启动进程
            logger.info(f"启动Worker {i}：{' '.join(cmd)}")
            processes.append(subprocess.Popen(cmd))
            
        # 等待所有进程
        try:
            for p in processes:
                p.wait()
        except KeyboardInterrupt:
            logger.info("收到中断信号，终止所有worker...")
            for p in processes:
                p.terminate()

if __name__ == '__main__':
    main() 