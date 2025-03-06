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
import glob

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
                    response = self.load_model(message.get('model_path', ''), message.get('gguf_path', ''))
                
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
            # 将数据转换为可序列化的格式
            if isinstance(data, torch.Tensor):
                # 张量直接存储
                data_tensor = data.cpu()
            elif isinstance(data, dict):
                # 对于字典，使用JSON字符串
                import json
                data_str = json.dumps(data)
                data_tensor = torch.tensor([ord(c) for c in data_str], dtype=torch.int32, device='cpu')
            elif isinstance(data, (list, tuple)):
                # 对于列表或元组，直接转换为张量
                try:
                    data_tensor = torch.tensor(data, device='cpu')
                except:
                    # 如果无法直接转换，将其转为JSON字符串
                    import json
                    data_str = json.dumps(data)
                    data_tensor = torch.tensor([ord(c) for c in data_str], dtype=torch.int32, device='cpu')
            elif isinstance(data, str):
                # 字符串转换为ASCII码序列
                data_tensor = torch.tensor([ord(c) for c in data], dtype=torch.int32, device='cpu')
            else:
                # 其他类型转为字符串
                data_str = str(data)
                data_tensor = torch.tensor([ord(c) for c in data_str], dtype=torch.int32, device='cpu')
            
            # 使用dummy值作为v参数，简化兼容性
            dummy_tensor = torch.zeros(1, dtype=torch.float32, device='cpu')
            
            # 添加标记表示数据类型
            metadata = {
                'type': str(type(data).__name__),
                'is_tensor': isinstance(data, torch.Tensor),
                'is_dict': isinstance(data, dict),
                'is_list': isinstance(data, list),
                'is_string': isinstance(data, str)
            }
            
            # 创建metadata张量
            import json
            metadata_str = json.dumps(metadata)
            metadata_tensor = torch.tensor([ord(c) for c in metadata_str], dtype=torch.int32, device='cpu')
            
            # 存储元数据和数据
            key_meta = f"{key}_metadata"
            self.kvcache_client.store(batch_id, key_meta, metadata_tensor, dummy_tensor)
            self.kvcache_client.store(batch_id, key, data_tensor, dummy_tensor)
            
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
            # 先尝试检索元数据
            key_meta = f"{key}_metadata"
            metadata_tensor = self.kvcache_client.retrieve(batch_id, key_meta, device=device)
            
            # 检索实际数据
            data_tensor = self.kvcache_client.retrieve(batch_id, key, device=device)
            
            # 如果元数据不存在或检索失败
            if metadata_tensor is None:
                # 尝试直接返回数据
                return data_tensor
            
            # 解析元数据
            if isinstance(metadata_tensor, torch.Tensor):
                try:
                    metadata_str = ''.join(chr(int(i)) for i in metadata_tensor.tolist())
                    import json
                    metadata = json.loads(metadata_str)
                except:
                    # 无法解析元数据，返回原始数据
                    return data_tensor
            else:
                # 元数据不是张量，直接返回数据
                return data_tensor
            
            # 根据数据类型重建原始数据
            if metadata.get('is_tensor'):
                # 是张量，直接返回
                return data_tensor
            elif metadata.get('is_string'):
                # 字符串类型，将整数转回字符
                return ''.join(chr(int(i)) for i in data_tensor.tolist())
            elif metadata.get('is_dict') or metadata.get('is_list'):
                # 字典或列表，解析JSON
                try:
                    data_str = ''.join(chr(int(i)) for i in data_tensor.tolist())
                    import json
                    return json.loads(data_str)
                except:
                    # JSON解析失败，返回原始字符串
                    return data_str
            else:
                # 无法识别的类型或解析失败，返回原始数据
                return data_tensor
            
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
            input_tensor = torch.tensor(input_ids, device=device)
            
            # 生成参数
            max_new_tokens = input_data.get('max_new_tokens', 50)
            temperature = input_data.get('temperature', 0.7)
            top_p = input_data.get('top_p', 0.9)
            
            logger.info(f"Generation parameters: max_new_tokens={max_new_tokens}, "
                       f"temperature={temperature}, "
                       f"top_p={top_p}")
            
            # 开始生成
            logger.info("Starting actual text generation with model...")
            start_time = time.time()
            
            # 优化内存管理
            torch.cuda.empty_cache()
            
            with torch.no_grad():
                # 检查模型是否已加载
                if not hasattr(self, 'model') or self.model is None:
                    raise ValueError("模型未正确加载，无法进行推理")
                    
                # 设置生成参数
                gen_kwargs = {
                    'max_new_tokens': max_new_tokens,
                    'do_sample': True,
                    'temperature': temperature,
                    'top_p': top_p
                }
                
                # 使用模型进行实际生成
                logger.info("正在使用模型生成文本...")
                output = self.model.generate(
                    inputs=input_tensor.unsqueeze(0),  # 添加批次维度
                    **gen_kwargs
                )
                
                # 获取生成的token ids
                output_token_ids = output[0].cpu().tolist()
                logger.info(f"模型生成了{len(output_token_ids)-len(input_ids)}个新token")
                
            # 确保所有GPU操作完成
            torch.cuda.synchronize()
            
            return {
                'status': 'success',
                'message': f"Worker {self.worker_id} successfully generated text",
                'time_taken': time.time() - start_time,
                'output_ids': output_token_ids
            }
            
        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
            return {
                'status': 'error',
                'error': str(e),
                'output_ids': []
            }

    def load_model(self, model_path: str, gguf_path: str) -> Dict[str, Any]:
        """加载模型"""
        try:
            logger.info(f"开始加载模型: {model_path}")
            
            # 先验证路径
            if not model_path:
                logger.error("模型路径为空")
                return {'status': 'error', 'error': '模型路径不能为空'}
            
            # 加载tokenizer
            logger.info("加载tokenizer...")
            try:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                logger.info("Tokenizer加载成功")
            except Exception as e:
                logger.error(f"加载tokenizer出错: {str(e)}")
                return {'status': 'error', 'error': f"加载tokenizer失败: {str(e)}"}
            
            # 加载模型配置
            logger.info("加载模型配置...")
            try:
                from transformers import AutoConfig
                self.config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                # 设置默认数据类型
                torch.set_default_dtype(self.config.torch_dtype)
                logger.info(f"模型配置加载成功，类型: {self.config.model_type}")
            except Exception as e:
                logger.error(f"加载模型配置出错: {str(e)}")
                return {'status': 'error', 'error': f"加载模型配置失败: {str(e)}"}
            
            # 检测是否有FALLBACK_MODE环境变量，用于在GGUF加载失败时回退
            fallback_mode = os.environ.get('KTRANSFORMERS_FALLBACK', 'FALSE').upper() == 'TRUE'
            
            logger.info("加载模型权重...")
            try:
                # 尝试加载模型
                if fallback_mode:
                    # 简单加载模式 - 不使用GGUF，直接从HuggingFace加载
                    logger.info("使用简单加载模式 (FALLBACK MODE)...")
                    from transformers import AutoModelForCausalLM
                    
                    # 确保CUDA可用
                    if torch.cuda.is_available():
                        device_map = "auto"
                    else:
                        device_map = "cpu"
                        logger.warning("CUDA不可用，使用CPU加载模型")
                    
                    # 直接加载原始模型
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        config=self.config,
                        trust_remote_code=True,
                        torch_dtype=self.config.torch_dtype,
                        device_map=device_map
                    )
                    logger.info("使用简单模式成功加载模型")
                else:
                    # 标准GGUF加载模式
                    try:
                        # 自定义模型处理
                        with torch.device("meta"):
                            from transformers import AutoModelForCausalLM
                            
                            # 检查是否是自定义模型类型
                            if hasattr(self.config, 'architectures') and len(self.config.architectures) > 0:
                                arch = self.config.architectures[0]
                                logger.info(f"模型架构: {arch}")
                                
                                # 加载自定义模型类
                                if "Llama" in arch:
                                    from ktransformers.models.modeling_llama import LlamaForCausalLM
                                    self.config._attn_implementation = "eager"
                                    self.model = LlamaForCausalLM(self.config)
                                elif "DeepseekV2" in arch:
                                    from ktransformers.models.modeling_deepseek import DeepseekV2ForCausalLM
                                    self.model = DeepseekV2ForCausalLM(self.config)
                                elif "DeepseekV3" in arch:
                                    from ktransformers.models.modeling_deepseek_v3 import DeepseekV3ForCausalLM
                                    self.model = DeepseekV3ForCausalLM(self.config)  
                                elif "Qwen2Moe" in arch:
                                    from ktransformers.models.modeling_qwen2_moe import Qwen2MoeForCausalLM
                                    self.config._attn_implementation = "flash_attention_2"
                                    self.model = Qwen2MoeForCausalLM(self.config)
                                elif "Mixtral" in arch:
                                    from ktransformers.models.modeling_mixtral import MixtralForCausalLM
                                    self.config._attn_implementation = "flash_attention_2"
                                    self.model = MixtralForCausalLM(self.config)
                                else:
                                    # 默认模型
                                    logger.info(f"使用默认AutoModelForCausalLM加载: {arch}")
                                    self.model = AutoModelForCausalLM.from_config(
                                        self.config, trust_remote_code=True, attn_implementation="flash_attention_2"
                                    )
                            else:
                                # 没有指定架构，使用默认模型
                                logger.info("未找到模型架构，使用默认AutoModelForCausalLM")
                                self.model = AutoModelForCausalLM.from_config(
                                    self.config, trust_remote_code=True
                                )
                        
                        # 从这里开始尝试使用optimize_and_load_gguf
                        logger.info(f"尝试使用optimize_and_load_gguf加载模型: {gguf_path}")
                        
                        # 获取优化规则路径
                        from ktransformers.optimize.optimize import optimize_and_load_gguf
                        
                        # 简化规则路径查找
                        current_dir = os.path.dirname(os.path.abspath(__file__))
                        ktransformer_rules_dir = os.path.join(current_dir, "..", "ktransformers", "optimize", "optimize_rules")
                        if not os.path.exists(ktransformer_rules_dir):
                            # 尝试其他可能的路径
                            ktransformer_rules_dir = os.path.join(current_dir, "optimize", "optimize_rules")
                        
                        logger.info(f"优化规则目录: {ktransformer_rules_dir}")
                        
                        # 设置超时，防止无限等待
                        import signal
                        
                        def timeout_handler(signum, frame):
                            raise TimeoutError("优化和加载GGUF模型超时")
                        
                        # 设置30秒超时
                        signal.signal(signal.SIGALRM, timeout_handler)
                        signal.alarm(30)  # 30秒超时
                        
                        try:
                            # 确保有合适的优化规则文件
                            # 查找可能的规则文件
                            arch_name = self.config.architectures[0] if hasattr(self.config, 'architectures') and self.config.architectures else "unknown"
                            rule_files = glob.glob(os.path.join(ktransformer_rules_dir, "*.yaml"))
                            
                            # 尝试找到匹配的规则文件
                            optimize_config_path = None
                            if rule_files:
                                # 有规则文件，选择匹配度最高的
                                for rule_file in rule_files:
                                    rule_name = os.path.basename(rule_file)
                                    # 尝试根据模型类型匹配规则文件
                                    if arch_name in rule_name:
                                        optimize_config_path = rule_file
                                        logger.info(f"找到匹配的规则文件: {rule_name}")
                                        break
                                
                                # 如果没有匹配的，使用第一个作为默认
                                if optimize_config_path is None:
                                    optimize_config_path = rule_files[0]
                                    logger.info(f"使用默认规则文件: {os.path.basename(optimize_config_path)}")
                            else:
                                # 没有找到规则文件，改用简单加载模式
                                logger.error("找不到优化规则文件，切换到简单加载模式")
                                signal.alarm(0)  # 取消超时
                                os.environ['KTRANSFORMERS_FALLBACK'] = 'TRUE'
                                return self.load_model(model_path, gguf_path)
                            
                            # 执行optimize_and_load_gguf，现在确保规则文件存在
                            logger.info(f"使用规则文件: {optimize_config_path}")
                            optimize_and_load_gguf(self.model, optimize_config_path, gguf_path, self.config)
                            # 取消超时
                            signal.alarm(0)
                        except TimeoutError:
                            logger.error("GGUF加载超时，切换到简单加载模式")
                            signal.alarm(0)  # 确保取消超时
                            # 递归调用自身，但启用fallback模式
                            os.environ['KTRANSFORMERS_FALLBACK'] = 'TRUE'
                            return self.load_model(model_path, gguf_path)
                            
                    except Exception as e:
                        logger.error(f"尝试使用optimize_and_load_gguf失败: {str(e)}")
                        import traceback
                        logger.error(traceback.format_exc())
                        
                        # 切换到简单加载模式
                        logger.info("切换到简单加载模式...")
                        os.environ['KTRANSFORMERS_FALLBACK'] = 'TRUE'
                        return self.load_model(model_path, gguf_path)
                    
                # 设置为评估模式
                self.model.eval()
                logger.info("模型加载成功，设置为评估模式")
                
                # 加载生成配置
                try:
                    from transformers import GenerationConfig
                    self.model.generation_config = GenerationConfig.from_pretrained(model_path)
                    logger.info("生成配置加载成功")
                except Exception as e:
                    logger.warning(f"无法加载生成配置，使用默认配置: {str(e)}")
                    # 创建默认生成配置
                    from transformers import GenerationConfig
                    gen_config = GenerationConfig(
                        temperature=0.7,
                        top_p=0.9,
                        do_sample=True
                    )
                    self.model.generation_config = gen_config
                    
                # 确保pad_token_id设置
                if self.model.generation_config.pad_token_id is None:
                    self.model.generation_config.pad_token_id = self.model.generation_config.eos_token_id
                    
                logger.info("模型和相关配置加载完成")
                return {'status': 'success', 'message': '模型加载成功'}
                
            except Exception as e:
                logger.error(f"加载模型出错: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                return {'status': 'error', 'error': f"加载模型失败: {str(e)}"}
                
        except Exception as e:
            logger.error(f"模型加载过程中出现未知错误: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {'status': 'error', 'error': f"未知错误: {str(e)}"}

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