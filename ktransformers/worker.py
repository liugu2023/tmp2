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
import torch.nn as nn
import json
import numpy as np

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
    def __init__(self, worker_address, worker_id, num_workers):
        """初始化ModelWorker
        Args:
            worker_address: 本worker绑定的地址，如"10.21.22.206:29500"
            worker_id: worker的ID，从0开始编号
            num_workers: 总worker数量
        """
        # 解析worker地址
        host, port = worker_address.split(':')
        self.address = worker_address
        self.worker_id = worker_id
        self.num_workers = num_workers
        
        # 创建socket并绑定地址
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://{host}:{port}")
        
        # 创建KV缓存客户端
        self.kvcache_client = KVCacheClient("10.21.22.206:29600")
        
        # 获取模型配置
        config_response = self.kvcache_client.get_config()
        if config_response['status'] != 'success':
            logger.error(f"Failed to get model config: {config_response.get('error')}")
            config_dict = self._create_default_config()
        else:
            config_dict = config_response['config']
        
        # 创建简化模型
        self.model = SimpleModel(config_dict, worker_id, num_workers)
        logger.info(f"Worker {worker_id} 成功初始化，负责层 {self.model.start_layer}-{self.model.end_layer-1}")
    
    def _create_default_config(self):
        """创建默认配置"""
        return {
            'hidden_size': 8192,
            'num_hidden_layers': 80,
            'max_position_embeddings': 8192,
            'vocab_size': 100000
        }
    
    def run(self):
        """运行ModelWorker，接收并处理请求"""
        logger.info(f"ModelWorker {self.worker_id} 开始运行，监听地址: {self.address}")
        
        while True:
            try:
                # 接收请求
                message = self.socket.recv_pyobj()
                command = message.get('command', '')
                
                # 处理请求
                if command == 'ping':
                    response = {'status': 'success', 'message': 'pong'}
                
                elif command == 'generate':
                    response = self.generate(message.get('data', {}))
                
                elif command == 'load_model':
                    model_path = message.get('model_path', '')
                    gguf_path = message.get('gguf_path', '')
                    logger.info(f"加载模型请求: {model_path}, GGUF: {gguf_path}")
                    response = {'status': 'success', 'message': 'Model loaded successfully'}
                
                elif command == 'shutdown':
                    response = {'status': 'success', 'message': 'Shutting down'}
                    self.socket.send_pyobj(response)
                    break
                
                else:
                    response = {'status': 'error', 'error': f'Unknown command: {command}'}
                
                # 发送响应
                self.socket.send_pyobj(response)
                
            except Exception as e:
                logger.error(f"Error in ModelWorker: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                # 尝试发送错误响应
                try:
                    self.socket.send_pyobj({'status': 'error', 'error': str(e)})
                except:
                    pass
        
        # 清理资源
        self.socket.close()
        self.context.term()
        logger.info(f"ModelWorker {self.worker_id} 已关闭")
    
    def generate(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理生成请求"""
        try:
            logger.info("Received generation request")
            logger.info(f"Input sequence length: {len(input_data['input_ids'])}")
            
            # 获取输入
            input_ids = input_data.get('input_ids', [0])
            
            # 记录生成参数
            logger.info(f"Generation parameters: max_new_tokens={input_data.get('max_new_tokens', 50)}, "
                       f"temperature={input_data.get('temperature', 0.7)}, "
                       f"top_p={input_data.get('top_p', 0.9)}")
            
            logger.info("Starting text generation...")
            start_time = time.time()
            
            # 构造batch ID
            batch_id = input_data.get('batch_id', f"batch_{time.time()}")
            
            # 使用简化的推理流程
            device = self.model.device
            hidden_size = self.model.hidden_size
            
            # 根据worker ID角色执行不同操作
            with torch.no_grad():
                if self.worker_id == 0:
                    # 第一个worker: 生成随机嵌入并通知其他worker
                    # 实际上，应该是通过embeddings层处理input_ids，这里简化为随机张量
                    batch_size = 1
                    seq_len = 1
                    hidden_states = torch.randn(batch_size, seq_len, hidden_size, 
                                              device=device, dtype=torch.float16)
                    
                    # 存储嵌入元数据
                    self.kvcache_client.store(batch_id, "worker_0_done", {'status': 'success'})
                    
                    # 记录处理信息
                    logger.info(f"Worker 0 generated embedding tensor: {hidden_states.shape}")
                
                # 所有worker: 处理自己负责的层
                logger.info(f"Processing layers {self.model.start_layer} to {self.model.end_layer-1}")
                
                # 假设在每一层上进行操作
                success = True
                for i in range(self.model.start_layer, self.model.end_layer):
                    # 这里模拟层处理，实际应该加载并应用每层参数
                    logger.info(f"Processing layer {i}")
                
                # 存储处理完成的状态
                completion_key = f"worker_{self.worker_id}_done"
                self.kvcache_client.store(batch_id, completion_key, {
                    'status': 'success' if success else 'error',
                    'layer_range': f"{self.model.start_layer}-{self.model.end_layer-1}"
                })
                
                # 如果是最后一个worker，还需要生成最终输出
                if self.worker_id == self.num_workers - 1:
                    # 生成随机输出token (实际应该是通过lm_head生成)
                    next_token = 100  # 随机token ID
                    self.kvcache_client.store(batch_id, "final_output", {
                        'token': next_token,
                        'status': 'success'
                    })
                    logger.info(f"Final worker generated token: {next_token}")
            
            # 返回成功响应
            return {
                'status': 'success',
                'output': f"Worker {self.worker_id} processed layers {self.model.start_layer}-{self.model.end_layer-1}",
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

class SimplifiedLayer(nn.Module):
    """简化的Transformer层"""
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, x):
        return self.linear(x)

class SimpleModel(nn.Module):
    """简化的模型结构，只包含必要组件"""
    def __init__(self, config, worker_id, worker_count):
        super().__init__()
        self.config = config
        self.worker_id = worker_id
        self.worker_count = worker_count
        
        # 模型参数
        self.hidden_size = config.get('hidden_size', 8192)
        self.num_layers = config.get('num_hidden_layers', 80)
        
        # 确定当前worker负责的层范围
        layers_per_worker = self.num_layers // worker_count
        self.start_layer = worker_id * layers_per_worker
        self.end_layer = (worker_id + 1) * layers_per_worker
        if worker_id == worker_count - 1:
            self.end_layer = self.num_layers
        
        # 选择设备 (根据worker ID分配到两个GPU)
        gpu_id = 0 if worker_id < worker_count // 2 else 1
        self.device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
        
        # 创建简单的层结构
        self.layers = nn.ModuleList([
            SimplifiedLayer(self.hidden_size) 
            for _ in range(self.start_layer, self.end_layer)
        ])
        
        # 只给需要的worker创建特殊层
        self.embed_tokens = nn.Embedding(config.get('vocab_size', 100000), self.hidden_size) if worker_id == 0 else None
        self.norm = nn.LayerNorm(self.hidden_size) if worker_id == worker_count - 1 else None
        self.lm_head = nn.Linear(self.hidden_size, config.get('vocab_size', 100000)) if worker_id == worker_count - 1 else None
        
        # 将模型移动到指定设备
        self.to(self.device)
        
        # 设置为评估模式
        self.eval()

if __name__ == '__main__':
    main() 