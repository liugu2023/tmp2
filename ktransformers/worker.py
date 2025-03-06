import os
import torch
import zmq
import pickle
from typing import Dict, Any
import logging
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, GenerationConfig, TextStreamer
import time
import subprocess
import multiprocessing
from ktransformers.kvcache_client import KVCacheClient

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

def start_worker_process(worker_id: int, base_port: int, num_workers: int):
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
    
    logger.info(f"Starting worker {worker_id} on NUMA node {node_id} with port {port}")
    subprocess.Popen(cmd)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_port', type=int, default=29500)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--worker_address', type=str, required=True,
                        help='Worker address in format host:port')
    parser.add_argument('--worker_id', type=int, default=0,
                        help='Worker ID (0-7)')
    args = parser.parse_args()
    
    # 创建并运行worker
    worker = ModelWorker(args.worker_address, args.worker_id, args.num_workers)
    worker.run()

def load_model_and_tokenizer(model_path: str, gguf_path: str):
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    logger.info("Loading model config...")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    torch.set_default_dtype(config.torch_dtype)
    
    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        trust_remote_code=True,
        device_map="auto"
    )
    model.eval()
    
    return model, tokenizer

class ModelWorker:
    def __init__(self, address: str, worker_id: int = 0, num_workers: int = 1, shared_components=None):
        self.worker_id = worker_id
        self.num_workers = num_workers
        self.shared_components = shared_components
        
        # 设置 GPU 设备
        if torch.cuda.is_available():
            # 根据 worker_id 分配 GPU
            gpu_id = worker_id % torch.cuda.device_count()
            torch.cuda.set_device(gpu_id)
            logger.info(f"Worker {worker_id} using GPU {gpu_id}")
        
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        host, port = address.split(':')
        self.socket.bind(f"tcp://0.0.0.0:{port}")
        self.socket.setsockopt(zmq.RCVHWM, 0)
        self.socket.setsockopt(zmq.SNDHWM, 0)
        
        self.model = None
        self.tokenizer = None
        logger.info(f"Worker {worker_id} started at {address}, listening on all interfaces")

    def load_model(self, model_path: str, gguf_path: str):
        # 创建KVCache客户端
        self.kvcache_client = KVCacheClient(f"10.21.22.206:29600")
        
        # 获取模型配置
        logger.info("从KVCache服务器获取模型配置")
        response = self.kvcache_client.get_config()
        if response['status'] != 'success':
            logger.error("获取模型配置失败")
            raise RuntimeError("Failed to get model config")
        
        # 从字典创建配置对象
        config_dict = response['config']
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        # 更新配置
        for key, value in config_dict.items():
            if hasattr(config, key):
                if key == 'torch_dtype':
                    # 处理torch_dtype
                    if value == 'torch.float16':
                        setattr(config, key, torch.float16)
                    elif value == 'torch.float32':
                        setattr(config, key, torch.float32)
                    else:
                        setattr(config, key, torch.float32)  # 默认
                else:
                    setattr(config, key, value)
        
        # 获取tokenizer
        logger.info("从KVCache服务器获取tokenizer")
        response = self.kvcache_client.get_tokenizer()
        if response['status'] != 'success':
            logger.error("获取tokenizer失败")
            raise RuntimeError("Failed to get tokenizer")
        
        self.tokenizer = response['tokenizer']
        
        # 计算每个进程应该处理的层
        total_layers = int(config_dict['num_hidden_layers'])  # 确保是整数
        layers_per_process = total_layers // self.num_workers
        start_layer = self.worker_id * layers_per_process
        
        # 修改这里的计算方式，避免使用条件表达式
        if self.worker_id < self.num_workers - 1:
            end_layer = start_layer + layers_per_process
        else:
            end_layer = total_layers
        
        # 确保所有数值都是整数
        start_layer = int(start_layer)
        end_layer = int(end_layer)
        
        # 动态计算所需显存
        hidden_size = int(config_dict['hidden_size'])  # 确保是整数
        # 假设最大上下文长度为8K
        context_len = 8192
        # 每层KV Cache = 2 * hidden_size * context_len * dtype_size (float16=2B)
        kv_size_per_layer = 2 * hidden_size * context_len * 2 / (1024**3)  # 单位：GB
        # 当前worker负责的层数
        my_layers = end_layer - start_layer
        # 预估KV缓存显存
        total_kv_cache = kv_size_per_layer * my_layers
        # 每层模型参数估算 (单位: GB)
        hidden_dim = hidden_size
        intermediate_dim = int(config_dict.get('intermediate_size', 4 * hidden_dim))  # 确保是整数
        param_size_per_layer = (12 * hidden_dim * hidden_dim + 4 * hidden_dim * intermediate_dim) * 2 / (1024**3)
        base_model_size = param_size_per_layer * my_layers
        # 工作空间估算 (1GB)
        workspace_size = 1.0
        # 最终显存预算
        final_memory_budget = total_kv_cache + base_model_size + workspace_size
        # 取整并加上安全余量
        gpu_mem = max(3, int(final_memory_budget) + 1)
        
        logger.info(f"Worker {self.worker_id} memory calculation:")
        logger.info(f"- Hidden size: {hidden_size}")
        logger.info(f"- Context length: {context_len}")
        logger.info(f"- KV cache per layer: {kv_size_per_layer:.2f} GB")
        logger.info(f"- Layer parameters: {param_size_per_layer:.2f} GB per layer")
        logger.info(f"- Handling {my_layers} layers: from {start_layer} to {end_layer-1}")
        logger.info(f"- Estimated KV cache: {total_kv_cache:.2f} GB")
        logger.info(f"- Estimated model parameters: {base_model_size:.2f} GB")
        logger.info(f"- Estimated workspace: {workspace_size:.2f} GB")
        logger.info(f"- Total GPU memory budget: {gpu_mem} GB")
        
        # 设置GPU ID
        gpu_id = 0 if self.worker_id < 4 else 1
        
        # 创建一个简化的模型结构
        # 这里使用torch.nn.Module来构建一个基本模型，
        # 它不包含实际权重，但提供generate()方法接口
        class SimpleModel(torch.nn.Module):
            def __init__(self, config, worker_id, worker_count, kvcache_client):
                super().__init__()
                self.config = config
                self.worker_id = worker_id
                self.worker_count = worker_count
                self.kvcache_client = kvcache_client
                
                # 添加必要的属性
                self.model = self
                self.device = torch.device(f'cuda:{0 if worker_id < 4 else 1}' if torch.cuda.is_available() else 'cpu')
                
                # 计算当前worker负责的层范围
                total_layers = config.num_hidden_layers
                layers_per_process = total_layers // worker_count
                self.start_layer = worker_id * layers_per_process
                self.end_layer = (self.start_layer + layers_per_process 
                                 if worker_id < worker_count - 1 
                                 else total_layers)
                
                # 创建精简的模型结构
                # 只保留必要的组件，其他组件设为None或空壳
                self.embed_tokens = None  # 词嵌入只在第一个worker上需要
                self.norm = None         # 最终归一化只在最后一个worker上需要
                self.lm_head = None      # 语言模型头只在最后一个worker上需要
                
                # 只创建当前worker负责的层
                self.layers = torch.nn.ModuleList([
                    torch.nn.Linear(config.hidden_size, config.hidden_size)
                    for _ in range(self.start_layer, self.end_layer)
                ])
                
                # 根据worker位置加载必要组件
                if worker_id == 0:
                    # 第一个worker需要词嵌入
                    logger.info("加载词嵌入层")
                    response = self.kvcache_client.get_shared_params()
                    if response['status'] == 'success' and 'params' in response:
                        params = response['params']
                        if params and 'embeddings' in params and params['embeddings'] is not None:
                            embeddings_tensor = params['embeddings']
                            # 确保是张量并且在正确的设备上
                            if isinstance(embeddings_tensor, torch.Tensor):
                                self.embed_tokens = torch.nn.Embedding.from_pretrained(
                                    embeddings_tensor.to(self.device),
                                    freeze=True
                                )
                                logger.info("词嵌入层加载成功")
                            else:
                                logger.warning("词嵌入参数不是张量类型")
                        else:
                            logger.warning("未找到词嵌入参数")
                    else:
                        logger.warning("获取共享参数失败或响应格式错误")
                
                if worker_id == worker_count - 1:
                    # 最后一个worker需要norm和lm_head
                    logger.info("加载norm和lm_head")
                    response = self.kvcache_client.get_shared_params()
                    if response['status'] == 'success' and 'params' in response:
                        params = response['params']
                        if params:
                            # 加载norm
                            if 'norm' in params and params['norm'] is not None:
                                norm_tensor = params['norm']
                                if isinstance(norm_tensor, torch.Tensor):
                                    self.norm = torch.nn.LayerNorm(config.hidden_size)
                                    self.norm.weight.data = norm_tensor.to(self.device)
                                    logger.info("norm层加载成功")
                                else:
                                    logger.warning("norm参数不是张量类型")
                            
                            # 加载lm_head
                            if 'lm_head' in params and params['lm_head'] is not None:
                                lm_head_tensor = params['lm_head']
                                if isinstance(lm_head_tensor, torch.Tensor):
                                    self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
                                    self.lm_head.weight.data = lm_head_tensor.to(self.device)
                                    logger.info("lm_head加载成功")
                                else:
                                    logger.warning("lm_head参数不是张量类型")
                    else:
                        logger.warning("获取共享参数失败或响应格式错误")
                
                # 创建生成配置
                self.generation_config = GenerationConfig(
                    temperature=0.6,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=config.pad_token_id or config.eos_token_id,
                    eos_token_id=config.eos_token_id,
                    max_new_tokens=100
                )
                
                logger.info(f"Worker {worker_id} 负责层 {self.start_layer}-{self.end_layer-1}")
            
            def load_layer_params(self, layer_idx):
                """按需加载层参数"""
                if layer_idx < self.start_layer or layer_idx >= self.end_layer:
                    return False
                    
                local_idx = layer_idx - self.start_layer
                response = self.kvcache_client.get_layer_params(layer_idx)
                if response['status'] != 'success':
                    return False
                    
                params = response['params']
                # 将参数加载到对应的层
                # 这里需要根据实际的层结构来设置参数
                # 当前是简化的实现
                return True
            
            def forward(self, hidden_states, attention_mask=None, **kwargs):
                """前向传播，只处理当前worker负责的层"""
                try:
                    # 确保输入在正确的设备上
                    hidden_states = hidden_states.to(self.device)
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(self.device)
                    
                    # 如果是第一个worker，需要处理词嵌入
                    if self.worker_id == 0 and self.embed_tokens is not None:
                        hidden_states = self.embed_tokens(hidden_states)
                    
                    # 处理当前worker负责的层
                    for i, layer in enumerate(self.layers):
                        # 按需加载层参数
                        layer_idx = self.start_layer + i
                        self.load_layer_params(layer_idx)
                        # 确保层在正确的设备上
                        layer = layer.to(self.device)
                        # 处理这一层
                        hidden_states = layer(hidden_states)
                    
                    # 如果是最后一个worker，需要处理norm和lm_head
                    if self.worker_id == self.worker_count - 1:
                        if self.norm is not None:
                            self.norm = self.norm.to(self.device)
                            hidden_states = self.norm(hidden_states)
                        if self.lm_head is not None:
                            self.lm_head = self.lm_head.to(self.device)
                            hidden_states = self.lm_head(hidden_states)
                    
                    return hidden_states
                
                except Exception as e:
                    logger.error(f"Forward pass error: {str(e)}")
                    import traceback
                    logger.error(traceback.format_exc())
                    raise
        
        # 创建简化模型实例
        self.model = SimpleModel(config, self.worker_id, self.num_workers, self.kvcache_client)
        
        # 设置模型准备好的标志
        self.model_ready = True
        
        logger.info(f"Worker {self.worker_id} 成功初始化，负责层 {start_layer}-{end_layer-1}")
        return True

    # 修改 CustomStreamer 类定义
    class CustomStreamer(TextStreamer):
        def __init__(self, tokenizer, on_text_chunk=None, **kwargs):
            super().__init__(tokenizer, **kwargs)
            self.on_text_chunk = on_text_chunk

        def on_finalized_text(self, text: str, stream_end: bool = False):
            if self.on_text_chunk:
                self.on_text_chunk(text, stream_end)

    def generate(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            logger.info("Received generation request")
            logger.info(f"Input sequence length: {len(input_data['input_ids'])}")
            
            # 创建输入张量并设置正确的数据类型
            logger.info("Moving input tensors to GPU...")
            device = self.model.device
            dtype = torch.float16  # 使用与模型相同的数据类型
            
            # 确保输入张量使用正确的数据类型
            input_ids = torch.tensor(input_data['input_ids'], device=device)
            attention_mask = torch.tensor(input_data['attention_mask'], device=device)
            
            # 初始化 hidden_states，确保在所有条件分支前都已定义
            hidden_states = input_ids  # 默认初始化
            
            # 生成参数日志
            max_new_tokens = input_data.get('max_new_tokens', 512)
            temperature = input_data.get('temperature', 0.6)
            top_p = input_data.get('top_p', 0.9)
            logger.info(f"Generation parameters: max_new_tokens={max_new_tokens}, "
                       f"temperature={temperature}, "
                       f"top_p={top_p}")
            
            # 生成文本
            logger.info("Starting text generation...")
            start_time = time.time()
            
            # 创建 KV Cache 客户端
            kvcache_client = KVCacheClient("10.21.22.206:29600")
            
            # 生成唯一的 batch ID
            batch_id = f"{time.time()}_{self.worker_id}"
            
            with torch.no_grad(), torch.amp.autocast('cuda'):
                # 如果是第一个worker，需要处理词嵌入
                if self.worker_id == 0:
                    if self.model.embed_tokens is not None:
                        # 将输入转换为正确的数据类型
                        hidden_states = self.model.embed_tokens(input_ids)
                        hidden_states = hidden_states.to(dtype)
                        # 存储嵌入结果供其他worker使用
                        cpu_hidden_states = hidden_states.cpu().numpy()  # 使用numpy而不是tolist
                        kvcache_client.store(batch_id, "embeddings", cpu_hidden_states)
                        logger.info(f"Stored embeddings with shape {hidden_states.shape}")
                else:
                    # 其他worker需要等待并获取嵌入结果
                    result = None
                    for _ in range(10):  # 最多尝试10次
                        try:
                            result = kvcache_client.retrieve(batch_id, "embeddings", device=device)
                            if result is not None:
                                # 将numpy数组转换回张量，并设置正确的设备和数据类型
                                hidden_states = torch.from_numpy(result).to(device=device, dtype=dtype)
                                logger.info(f"Retrieved embeddings with shape {hidden_states.shape}")
                                break
                        except Exception as e:
                            logger.warning(f"Retrieval attempt failed: {str(e)}")
                        time.sleep(0.1)
                    
                    if result is None:
                        logger.error("Failed to retrieve embeddings after multiple attempts")
                        raise RuntimeError("Failed to retrieve embeddings from first worker")
                
                # 确保 hidden_states 在这一点上已经正确初始化
                if hidden_states is None:
                    raise RuntimeError("hidden_states is None after initialization phase")
                
                # 处理当前worker负责的层
                for layer_idx in range(self.model.start_layer, self.model.end_layer):
                    # 确保输入在正确的设备和数据类型上
                    hidden_states = hidden_states.to(device).to(dtype)
                    # 获取当前层
                    layer = self.model.layers[layer_idx - self.model.start_layer].to(device)
                    # 记录形状以便调试
                    logger.info(f"Layer {layer_idx} input shape: {hidden_states.shape}")
                    # 处理这一层
                    hidden_states = layer(hidden_states)
                    # 存储结果
                    cpu_hidden_states = hidden_states.cpu().numpy()
                    kvcache_client.store(batch_id, f"layer_{layer_idx}", cpu_hidden_states)
                    logger.info(f"Processed and stored layer {layer_idx} with shape {hidden_states.shape}")
                
                # 如果是最后一个worker，需要处理norm和lm_head
                if self.worker_id == self.num_workers - 1:
                    if self.model.norm is not None:
                        hidden_states = self.model.norm(hidden_states.to(device).to(dtype))
                    if self.model.lm_head is not None:
                        hidden_states = self.model.lm_head(hidden_states)
                    # 存储最终结果
                    cpu_hidden_states = hidden_states.cpu().numpy()
                    kvcache_client.store(batch_id, "final", cpu_hidden_states)
                    logger.info(f"Stored final output with shape {hidden_states.shape}")
                
                # 等待最终结果
                if self.worker_id != self.num_workers - 1:
                    result = None
                    for _ in range(10):  # 最多尝试10次
                        try:
                            result = kvcache_client.retrieve(batch_id, "final", device=device)
                            if result is not None:
                                hidden_states = torch.from_numpy(result).to(device=device, dtype=dtype)
                                logger.info(f"Retrieved final output with shape {hidden_states.shape}")
                                break
                        except Exception as e:
                            logger.warning(f"Final result retrieval attempt failed: {str(e)}")
                        time.sleep(0.1)
                    
                    if result is None:
                        logger.error("Failed to retrieve final output after multiple attempts")
                        raise RuntimeError("Failed to retrieve final output from last worker")
                
                # 生成完成后清理缓存
                kvcache_client.clear(batch_id)
                
                # 确保输出在CPU上，以便序列化
                hidden_states = hidden_states.cpu().numpy()
                
                # 返回结果
                return {
                    'status': 'success',
                    'output': hidden_states.tolist(),
                    'time_taken': time.time() - start_time
                }
            
        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
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
            logger.info("Worker shutdown complete")

if __name__ == '__main__':
    main() 