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
    parser.add_argument('--worker_address', type=str, required=False)
    parser.add_argument('--worker_id', type=int, default=0)
    args = parser.parse_args()
    
    if args.worker_address:
        # 单个 worker 进程模式
        worker = ModelWorker(args.worker_address, args.worker_id, args.num_workers)
        worker.run()
    else:
        # 主进程模式，启动多个 worker
        for worker_id in range(args.num_workers):
            start_worker_process(worker_id, args.base_port, args.num_workers)
        
        # 等待所有进程
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down all workers...")

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
        # 如果有共享组件，使用共享组件
        if self.shared_components:
            logger.info("Using shared model components...")
            config = self.shared_components['config']
            self.tokenizer = AutoTokenizer.from_pretrained(self.shared_components['tokenizer_path'], trust_remote_code=True)
        else:
            # 否则正常加载
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            
            logger.info("Loading model config...")
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        
        torch.set_default_dtype(config.torch_dtype)
        
        # 计算每个进程应该处理的层
        total_layers = config.num_hidden_layers
        layers_per_process = total_layers // self.num_workers
        start_layer = self.worker_id * layers_per_process
        end_layer = start_layer + layers_per_process if self.worker_id < self.num_workers - 1 else total_layers
        
        # 动态计算所需显存
        hidden_size = config.hidden_size
        # 假设最大上下文长度为8K
        context_len = 8192
        # 每层KV Cache = 2 * hidden_size * context_len * dtype_size (float16=2B)
        kv_size_per_layer = 2 * hidden_size * context_len * 2 / (1024**3)  # 单位：GB
        # 当前worker负责的层数
        my_layers = end_layer - start_layer
        # 预估KV缓存显存
        total_kv_cache = kv_size_per_layer * my_layers
        # 每层模型参数估算 (单位: GB)
        hidden_dim = config.hidden_size
        intermediate_dim = config.intermediate_size if hasattr(config, 'intermediate_size') else 4 * hidden_dim
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
        
        # 设置内存分配
        max_memory = {
            0: f"{gpu_mem}GiB" if self.worker_id < 4 else None,    # 前4个进程使用 GPU 0
            1: f"{gpu_mem}GiB" if self.worker_id >= 4 else None,   # 后4个进程使用 GPU 1
            'cpu': "138GiB"  # CPU 内存
        }
        
        # 创建一个局部的设备映射
        device_map = {}
        
        # 如果使用共享组件，不需要加载embed_tokens
        if not self.shared_components:
            device_map['model.embed_tokens'] = 'cpu'
            device_map['lm_head'] = 'cpu'
        
        device_map['model.norm'] = 'cpu'
        
        # 分配层到对应的 GPU
        gpu_id = 0 if self.worker_id < 4 else 1
        for i in range(start_layer, end_layer):
            device_map[f'model.layers.{i}'] = f'cuda:{gpu_id}'
        
        # 创建模型 - 只加载需要的层
        if self.shared_components:
            logger.info("Creating model with shared components...")
            # 创建一个简化的模型结构
            from transformers.modeling_utils import PreTrainedModel
            
            class PartialModel(PreTrainedModel):
                def __init__(self, config):
                    super().__init__(config)
                    self.config = config
                    
                # 其他必要的方法...
            
            self.model = PartialModel(config)
            
            # 添加共享的embedding层
            self.model.model.embed_tokens = self.shared_components['embed_tokens']
            self.model.lm_head = self.shared_components['lm_head']
            
            # 只加载本worker需要处理的层
            for i in range(start_layer, end_layer):
                layer_key = f'model.layers.{i}'
                # 检查该层是否已经在另一个进程中加载
                if layer_key in self.shared_components['loaded_layers']:
                    self.model.model.layers[i] = self.shared_components['loaded_layers'][layer_key]
                else:
                    # 加载新层
                    layer = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        config=config,
                        device_map={f'model.layers.{i}': f'cuda:{gpu_id}'},
                        torch_dtype=torch.float16,
                    ).model.layers[i]
                    
                    self.model.model.layers[i] = layer
                    # 将层添加到共享字典
                    self.shared_components['loaded_layers'][layer_key] = layer
        else:
            # 使用标准方法加载
            model_kwargs = {
                "device_map": device_map,
                "max_memory": max_memory,
                "torch_dtype": torch.float16,
                "low_cpu_mem_usage": True
            }
            
            logger.info("Applying optimization configurations...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                config=config,
                trust_remote_code=True,
                **model_kwargs
            )
        
        # 输出配置信息
        logger.info(f"Worker {self.worker_id} model configuration:")
        logger.info(f"- Handling layers: {start_layer} to {end_layer}")
        logger.info(f"- Using GPU: {gpu_id}")
        logger.info("- Device mapping:")
        for name, device in self.model.hf_device_map.items():
            if name.startswith(f'model.layers.{start_layer}'):
                logger.info(f"  {name}: {device}")
        
        self.model.eval()
        
        # 设置生成配置
        try:
            self.model.generation_config = GenerationConfig.from_pretrained(model_path)
        except Exception as e:
            logger.warning(f"Generation config can't auto create, making default. Message: {e}")
            self.model.generation_config = GenerationConfig(
                temperature=0.6,
                top_p=0.9,
                do_sample=True
            )
        if self.model.generation_config.pad_token_id is None:
            self.model.generation_config.pad_token_id = self.model.generation_config.eos_token_id
            
        logger.info("Model loaded successfully")

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
            
            # 创建输入张量
            logger.info("Moving input tensors to GPU...")
            input_ids = torch.tensor(input_data['input_ids'], device='cuda')
            attention_mask = torch.tensor(input_data['attention_mask'], device='cuda')
            
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
                # 在每一层处理完后，存储 KV Cache
                for i, layer in enumerate(self.model.model.layers):
                    if start_layer <= i < end_layer:
                        # 这是当前 worker 负责的层
                        k, v = layer.self_attn.compute_kv(input_ids, attention_mask)
                        kvcache_client.store(batch_id, i, k, v)
                    else:
                        # 需要从其他 worker 获取的层
                        k, v = kvcache_client.retrieve(batch_id, i, f"cuda:{gpu_id}")
                        if k is not None and v is not None:
                            # 使用获取的 KV Cache 继续处理
                            layer.self_attn.use_cached_kv(k, v)
                
                # 生成完成后清理缓存
                kvcache_client.clear(batch_id)
                
                # 创建自定义流式处理器
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
                
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=input_data.get('do_sample', True),
                    streamer=streamer
                )
                
                # 记录生成完成的信息
                final_time = time.time()
                total_gen_time = final_time - start_time
                logger.info("  - Generation completed:")
                logger.info(f"    - Total generation time: {total_gen_time:.2f} seconds")
                logger.info(f"    - Final sequence length: {len(outputs[0])} tokens")
                logger.info(f"    - New tokens generated: {len(outputs[0]) - len(input_ids[0])}")
                logger.info(f"    - Average speed: {(len(outputs[0]) - len(input_ids[0])) / total_gen_time:.2f} tokens/sec")
            
            end_time = time.time()
            generation_time = end_time - start_time
            tokens_generated = len(outputs[0]) - len(input_ids[0])
            tokens_per_second = tokens_generated / generation_time
            
            logger.info(f"Generation completed in {generation_time:.2f} seconds")
            logger.info(f"Tokens generated: {tokens_generated}")
            logger.info(f"Generation speed: {tokens_per_second:.2f} tokens/second")
            logger.info(f"Final sequence length: {len(outputs[0])}")
            
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
            logger.info("Worker shutdown complete")

if __name__ == '__main__':
    main() 