"""
嵌入层服务器
负责词表查询和嵌入层计算，减轻各worker负担
"""

import zmq
import torch
import logging
import argparse
from transformers import AutoTokenizer, AutoModel, AutoConfig
import time
import gc
from typing import Dict, Any, List, Optional
import os
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingServer:
    def __init__(self, server_address: str, model_path: str, skip_tokenizer: bool = True):
        # 初始化ZMQ上下文
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://{server_address}")
        self.socket.setsockopt(zmq.RCVHWM, 0)
        self.socket.setsockopt(zmq.SNDHWM, 0)
        logger.info(f"Embedding server listening at {server_address}")
        
        # 加载模型配置
        self.model_path = model_path
        logger.info(f"Loading model config from {model_path}")
        self.config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        
        # 如果需要tokenizer，才加载
        self.tokenizer = None
        if not skip_tokenizer:
            logger.info("Loading tokenizer")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        else:
            logger.info("Skipping tokenizer loading (will be handled by scheduler)")
        
        # 只加载嵌入层和最后的输出层
        logger.info("Loading embedding layer and output projection")
        self._load_embedding_layer()
        
        # 统计信息
        self.stats = {
            "requests_processed": 0,
            "tokens_processed": 0,
            "start_time": time.time()
        }
        
        logger.info("Embedding server initialization complete")
    
    def _load_embedding_layer(self):
        """只加载嵌入层和输出层，不加载完整模型"""
        try:
            logger.info("开始加载嵌入层和输出层...")
            
            # 创建模型参数匹配模式
            model_name = os.path.basename(self.model_path.rstrip("/"))
            logger.info(f"检测到模型: {model_name}")
            
            # 加载模型配置
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
            
            # 检测模型类型，确定参数路径模式
            if 'llama' in model_name.lower() or hasattr(config, 'model_type') and config.model_type == 'llama':
                embed_key = 'model.embed_tokens.weight'
                lm_head_key = 'lm_head.weight'
                logger.info("检测到LLaMA类型模型")
            elif 'mistral' in model_name.lower() or hasattr(config, 'model_type') and config.model_type == 'mistral':
                embed_key = 'model.embed_tokens.weight'
                lm_head_key = 'lm_head.weight'
                logger.info("检测到Mistral类型模型")
            elif 'bloom' in model_name.lower() or hasattr(config, 'model_type') and config.model_type == 'bloom':
                embed_key = 'transformer.word_embeddings.weight'
                lm_head_key = 'lm_head.weight'
                logger.info("检测到Bloom类型模型")
            else:
                # 默认为通用路径
                embed_key = 'model.embed_tokens.weight'
                lm_head_key = 'lm_head.weight'
                logger.info(f"未识别具体模型类型，使用默认路径模式: {embed_key}, {lm_head_key}")
            
            # 从模型检查点只加载词表相关参数
            import torch
            from pathlib import Path
            
            # 检查是否有safetensors格式文件
            from safetensors.torch import load_file as load_safetensors
            
            # 寻找权重文件
            weights_path = Path(self.model_path)
            safetensors_files = list(weights_path.glob("*.safetensors"))
            pytorch_files = list(weights_path.glob("pytorch_model*.bin"))
            
            # 创建空的嵌入层和LM头，稍后加载权重
            vocab_size = config.vocab_size
            hidden_size = config.hidden_size
            
            # 创建模型组件
            from torch import nn
            self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
            self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
            
            # 函数：从权重文件中加载特定层
            def find_and_load_tensor(file_path, key, safetensors_format=False):
                try:
                    if safetensors_format:
                        weights = load_safetensors(file_path)
                    else:
                        weights = torch.load(file_path, map_location='cpu')
                    
                    if key in weights:
                        logger.info(f"在 {file_path.name} 中找到 {key}")
                        return weights[key]
                    return None
                except Exception as e:
                    logger.warning(f"读取 {file_path} 出错: {e}")
                    return None
            
            # 加载嵌入层权重
            embed_loaded = False
            lm_head_loaded = False
            
            # 先尝试safetensors格式
            for sf in safetensors_files:
                if not embed_loaded:
                    embed_tensor = find_and_load_tensor(sf, embed_key, safetensors_format=True)
                    if embed_tensor is not None:
                        logger.info(f"从 {sf.name} 加载嵌入层, 大小: {embed_tensor.shape}")
                        self.embed_tokens.weight.data.copy_(embed_tensor)
                        embed_loaded = True
                
                if not lm_head_loaded:
                    lm_head_tensor = find_and_load_tensor(sf, lm_head_key, safetensors_format=True)
                    if lm_head_tensor is not None:
                        logger.info(f"从 {sf.name} 加载LM头, 大小: {lm_head_tensor.shape}")
                        self.lm_head.weight.data.copy_(lm_head_tensor)
                        lm_head_loaded = True
                
                if embed_loaded and lm_head_loaded:
                    break
            
            # 如果safetensors中未找到，尝试PyTorch格式
            if not (embed_loaded and lm_head_loaded):
                for pf in pytorch_files:
                    if not embed_loaded:
                        embed_tensor = find_and_load_tensor(pf, embed_key)
                        if embed_tensor is not None:
                            logger.info(f"从 {pf.name} 加载嵌入层, 大小: {embed_tensor.shape}")
                            self.embed_tokens.weight.data.copy_(embed_tensor)
                            embed_loaded = True
                    
                    if not lm_head_loaded:
                        lm_head_tensor = find_and_load_tensor(pf, lm_head_key)
                        if lm_head_tensor is not None:
                            logger.info(f"从 {pf.name} 加载LM头, 大小: {lm_head_tensor.shape}")
                            self.lm_head.weight.data.copy_(lm_head_tensor)
                            lm_head_loaded = True
                    
                    if embed_loaded and lm_head_loaded:
                        break
            
            # 检查是否成功加载
            if not embed_loaded:
                raise ValueError(f"未能找到嵌入层权重: {embed_key}")
            if not lm_head_loaded:
                raise ValueError(f"未能找到LM头权重: {lm_head_key}")
            
            # 将模型移动到GPU
            logger.info("将嵌入层和LM头移至CUDA设备")
            self.embed_tokens = self.embed_tokens.to("cuda:0")
            self.lm_head = self.lm_head.to("cuda:0")
            
            # 设置为评估模式
            self.embed_tokens.eval()
            self.lm_head.eval()
            
            # 显示内存使用情况
            logger.info(f"CUDA内存使用: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
            
            # 清理
            torch.cuda.empty_cache()
            gc.collect()
            
            logger.info("嵌入层和LM头加载完成")
            return True
        except Exception as e:
            logger.error(f"加载嵌入层出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def run(self):
        """运行嵌入服务器主循环"""
        logger.info("Embedding server ready to receive requests")
        try:
            while True:
                # 接收请求
                request = self.socket.recv_pyobj()
                command = request.get("command", "")
                
                if command == "shutdown":
                    logger.info("Received shutdown command")
                    self.socket.send_pyobj({"status": "success", "message": "Shutting down"})
                    break
                
                elif command == "stats":
                    self.socket.send_pyobj(self._get_stats())
                
                elif command == "tokenize":
                    # 处理文本->token IDs 的转换
                    result = self._handle_tokenize(request)
                    self.socket.send_pyobj(result)
                
                elif command == "embed_tokens":
                    # 处理token IDs->嵌入向量的转换
                    result = self._handle_embed_tokens(request)
                    self.socket.send_pyobj(result)
                
                elif command == "project_to_vocab":
                    # 处理隐藏状态->词汇表的映射
                    result = self._handle_projection(request)
                    self.socket.send_pyobj(result)
                
                else:
                    logger.warning(f"Unknown command: {command}")
                    self.socket.send_pyobj({
                        "status": "error",
                        "error": f"Unknown command: {command}"
                    })
                
                # 更新统计信息
                self.stats["requests_processed"] += 1
                
        except Exception as e:
            logger.error(f"Error in embedding server: {str(e)}")
        finally:
            logger.info("Shutting down embedding server")
            self.socket.close()
            self.context.term()
    
    def _handle_tokenize(self, request):
        """处理tokenize请求，将文本转换为token IDs"""
        try:
            text = request.get("text", "")
            add_special_tokens = request.get("add_special_tokens", True)
            
            if isinstance(text, str):
                # 单个文本
                encoding = self.tokenizer(
                    text, 
                    add_special_tokens=add_special_tokens, 
                    return_tensors="pt"
                )
                input_ids = encoding["input_ids"][0].tolist()
                attention_mask = encoding["attention_mask"][0].tolist()
            else:
                # 文本列表
                encoding = self.tokenizer(
                    text, 
                    add_special_tokens=add_special_tokens, 
                    padding=True,
                    return_tensors="pt"
                )
                input_ids = encoding["input_ids"].tolist()
                attention_mask = encoding["attention_mask"].tolist()
            
            # 更新统计
            if isinstance(input_ids[0], list):
                num_tokens = sum(len(ids) for ids in input_ids)
            else:
                num_tokens = len(input_ids)
            self.stats["tokens_processed"] += num_tokens
            
            return {
                "status": "success",
                "input_ids": input_ids,
                "attention_mask": attention_mask
            }
        except Exception as e:
            logger.error(f"Error in tokenize: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _handle_embed_tokens(self, request):
        """处理embedding请求，将token IDs转换为嵌入向量"""
        try:
            input_ids = torch.tensor(request.get("input_ids"), device="cuda:0")
            
            # 更新统计
            if input_ids.dim() > 1:
                num_tokens = input_ids.shape[0] * input_ids.shape[1]
            else:
                num_tokens = input_ids.shape[0]
            self.stats["tokens_processed"] += num_tokens
            
            # 执行嵌入
            with torch.no_grad():
                if input_ids.dim() == 1:
                    # 单个序列
                    embeddings = self.embed_tokens(input_ids)
                else:
                    # 批次处理
                    embeddings = self.embed_tokens(input_ids)
            
            # 转移到CPU并转换为numpy以便序列化
            embeddings_cpu = embeddings.cpu().numpy()
            
            return {
                "status": "success",
                "embeddings": embeddings_cpu
            }
        except Exception as e:
            logger.error(f"Error in embed_tokens: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _handle_projection(self, request):
        """处理投影请求，将隐藏状态映射到词汇表"""
        try:
            hidden_states = torch.tensor(request.get("hidden_states"), device="cuda:0")
            
            # 执行投影
            with torch.no_grad():
                logits = self.lm_head(hidden_states)
            
            # 转移到CPU并转换为numpy以便序列化
            logits_cpu = logits.cpu().numpy()
            
            return {
                "status": "success",
                "logits": logits_cpu
            }
        except Exception as e:
            logger.error(f"Error in project_to_vocab: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _get_stats(self):
        """获取服务器统计信息"""
        uptime = time.time() - self.stats["start_time"]
        
        # 计算速率
        tokens_per_second = self.stats["tokens_processed"] / uptime if uptime > 0 else 0
        
        return {
            "status": "success",
            "requests_processed": self.stats["requests_processed"],
            "tokens_processed": self.stats["tokens_processed"],
            "uptime_seconds": uptime,
            "tokens_per_second": tokens_per_second,
            "model_path": self.model_path
        }

def main():
    parser = argparse.ArgumentParser(description="Embedding Server")
    parser.add_argument("--server_address", type=str, default="0.0.0.0:29700",
                      help="Address to listen for requests")
    parser.add_argument("--model_path", type=str, required=True,
                      help="Path to the model")
    parser.add_argument("--skip_tokenizer", type=str, choices=["true", "false"], default="true",
                      help="Skip loading tokenizer (will be handled by scheduler)")
    parser.add_argument("--cuda_visible_devices", type=str, default="0",
                      help="Specify which CUDA devices to use, e.g. '0'")
    
    args = parser.parse_args()
    
    # 设置CUDA设备
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    
    # 输出CUDA信息
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        logger.info(f"CUDA current device: {torch.cuda.current_device()}")
        logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("CUDA not available, using CPU only!")
    
    # 启动嵌入服务器
    skip_tokenizer = args.skip_tokenizer.lower() == "true"
    try:
        server = EmbeddingServer(args.server_address, args.model_path, skip_tokenizer)
        server.run()
    except Exception as e:
        logger.error(f"Failed to start embedding server: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 