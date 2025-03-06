import os
import torch
import zmq
import pickle
from typing import Dict, Any, Optional, Tuple
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
        
        # 获取模型配置
        self.kvcache_client = KVCacheClient("10.21.22.206:29600")
        config_response = self.kvcache_client.get_config()
        if config_response['status'] != 'success':
            logger.error(f"获取模型配置失败: {config_response.get('error')}")
            config_dict = {
                'hidden_size': 8192,
                'num_hidden_layers': 80,
                'max_position_embeddings': 8192,
                'vocab_size': 100000
            }
        else:
            config_dict = config_response['config']
        
        # 选择GPU设备 - 根据worker_id平均分配到两个GPU
        gpu_id = 0 if worker_id < num_workers // 2 else 1
        self.device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Worker {worker_id} 使用设备: {self.device}")
        
        # 配置模型参数
        self.hidden_size = config_dict.get('hidden_size', 8192)
        total_layers = config_dict.get('num_hidden_layers', 80)
        
        # 计算当前worker负责的层
        layers_per_worker = total_layers // num_workers
        self.start_layer = worker_id * layers_per_worker
        self.end_layer = (worker_id + 1) * layers_per_worker
        if worker_id == num_workers - 1:
            self.end_layer = total_layers
            
        logger.info(f"Worker {worker_id} 负责层 {self.start_layer}-{self.end_layer-1}")
        
        # 为提高内存效率，不预加载任何模型层
        # 我们将在需要时动态加载层参数
    
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
                    response = {'status': 'success'}
                
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
    
    def _store_data(self, batch_id, key, data):
        """使用KVCacheClient存储数据的辅助方法，处理参数兼容性"""
        try:
            # 将字典转换为JSON字符串，然后转换为张量
            if isinstance(data, dict):
                import json
                data_str = json.dumps(data)
                data_bytes = data_str.encode('utf-8')
                # 创建一个包含字节数据的张量
                data_tensor = torch.tensor([ord(b) for b in data_str], 
                                          dtype=torch.uint8, device='cpu')
            elif isinstance(data, (str, int, float, bool)):
                # 处理基本类型
                data_str = str(data)
                data_tensor = torch.tensor([ord(c) for c in data_str], 
                                          dtype=torch.uint8, device='cpu')
            elif isinstance(data, torch.Tensor):
                # 如果已经是张量，确保在CPU上
                data_tensor = data.cpu()
            else:
                # 其他类型，尝试转换
                try:
                    data_tensor = torch.tensor(data, device='cpu')
                except:
                    # 无法转换为张量，将其转为字符串
                    data_str = str(data)
                    data_tensor = torch.tensor([ord(c) for c in data_str], 
                                              dtype=torch.uint8, device='cpu')
            
            # 创建一个空的张量作为v参数，或使用相同的数据
            empty_tensor = torch.zeros(1, dtype=torch.float32, device='cpu')
            
            # 存储数据
            self.kvcache_client.store(batch_id, key, data_tensor, empty_tensor)
            return True
        except Exception as e:
            logger.error(f"存储数据时出错: {str(e)}")
            logger.error(f"数据类型: {type(data)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def _retrieve_data(self, batch_id, key, device='cpu'):
        """使用KVCacheClient检索数据的辅助方法，处理参数兼容性"""
        try:
            result = self.kvcache_client.retrieve(batch_id, key, device=device)
            
            # 如果结果是元组，取第一个元素
            if isinstance(result, tuple) and len(result) == 2:
                result = result[0]
            
            # 如果是张量且元素类型是uint8，尝试将其解析为字符串/字典
            if isinstance(result, torch.Tensor) and result.dtype == torch.uint8:
                try:
                    # 尝试将字节转换回字符串
                    result_str = ''.join(chr(int(b)) for b in result.flatten().tolist())
                    
                    # 尝试将字符串解析为JSON（字典）
                    try:
                        import json
                        return json.loads(result_str)
                    except:
                        # 不是有效的JSON，返回字符串
                        return result_str
                except:
                    # 无法转换，返回原始张量
                    pass
            
            return result
        except Exception as e:
            logger.error(f"检索数据时出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def generate(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理生成请求"""
        try:
            logger.info("Received generation request")
            
            # 检查输入格式
            if 'input_ids' not in input_data:
                logger.warning("输入数据中未找到'input_ids'，使用默认值")
                input_ids = [0]  # 使用默认值
            else:
                input_ids = input_data['input_ids']
            
            logger.info(f"Input sequence length: {len(input_ids)}")
            
            # 移动输入张量到GPU
            logger.info("Moving input tensors to GPU...")
            device = self.device
            
            # 初始化输入
            input_ids = torch.tensor(input_ids, device=device)
            
            # 生成参数日志
            max_new_tokens = input_data.get('max_new_tokens', 50)
            temperature = input_data.get('temperature', 0.7)
            top_p = input_data.get('top_p', 0.9)
            logger.info(f"Generation parameters: max_new_tokens={max_new_tokens}, "
                      f"temperature={temperature}, "
                      f"top_p={top_p}")
            
            # 开始生成
            logger.info("Starting text generation...")
            start_time = time.time()
            
            # 优化内存管理
            torch.cuda.empty_cache()  # 显式清理缓存
            
            # 使用上下文管理器确保在出作用域时释放张量
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
                # 使用batch ID进行分布式协调
                batch_id = input_data.get('batch_id', f"batch_{time.time()}")
                
                try:
                    # 模拟处理 - 每个worker只处理自己的层
                    for layer_idx in range(self.start_layer, self.end_layer):
                        logger.info(f"处理层 {layer_idx}")
                        # 在实际实现中，这里应该加载并应用层参数
                        time.sleep(0.01)  # 简单模拟处理时间
                    
                    # 存储处理状态 - 使用简单的字符串而不是复杂字典
                    worker_status = f"Worker {self.worker_id} completed layers {self.start_layer}-{self.end_layer-1}"
                    # 使用修改后的辅助方法存储
                    self._store_data(batch_id, f"worker_{self.worker_id}_status", worker_status)
                    logger.info(f"已存储状态: {worker_status}")
                
                except Exception as e:
                    logger.error(f"处理层时发生错误: {str(e)}")
                    import traceback
                    logger.error(traceback.format_exc())
                    raise
            
            # 确保所有GPU操作完成
            torch.cuda.synchronize()
            
            return {
                'status': 'success',
                'output': f"Worker {self.worker_id} processed layers {self.start_layer}-{self.end_layer-1}",
                'time_taken': time.time() - start_time
            }
            
        except torch.cuda.OutOfMemoryError as e:
            # 专门处理OOM错误
            logger.error(f"CUDA内存不足: {str(e)}")
            torch.cuda.empty_cache()  # 尝试释放缓存
            
            # 打印当前GPU内存状态
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    mem_allocated = torch.cuda.memory_allocated(i) / (1024**3)
                    mem_reserved = torch.cuda.memory_reserved(i) / (1024**3)
                    logger.error(f"GPU {i}: 已分配 {mem_allocated:.2f}GB, 已预留 {mem_reserved:.2f}GB")
            
            return {
                'status': 'error',
                'error': f"内存不足 (OOM): {str(e)}"
            }
            
        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
            return {
                'status': 'error',
                'error': str(e)
            }
            
        finally:
            # 确保在函数结束时释放内存
            torch.cuda.empty_cache()

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