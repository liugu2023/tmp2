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
        """只加载嵌入层和输出层"""
        try:
            logger.info("Loading embedding layers to CUDA...")
            
            # 使用更具体的加载方式，避免设备映射问题
            # 首先初始化一个空模型
            from transformers import AutoModelForCausalLM
            
            # 使用低级API直接加载模型
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                config=self.config,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map="cpu"  # 先加载到CPU
            )
            
            # 手动提取和移动嵌入层到GPU
            if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
                self.embed_tokens = model.model.embed_tokens.to("cuda:0")
                logger.info(f"Embedding loaded to cuda:0, dimension: {self.embed_tokens.embedding_dim}")
            else:
                logger.error("Could not find embedding layer in model")
                raise ValueError("Embedding layer not found")
            
            # 手动提取和移动LM头到GPU
            if hasattr(model, "lm_head"):
                self.lm_head = model.lm_head.to("cuda:0")
                logger.info(f"LM head loaded to cuda:0, shape: {self.lm_head.weight.shape}")
            else:
                logger.error("Could not find lm_head in model")
                raise ValueError("LM head not found")
            
            # 删除不需要的模型部分以节省内存
            del model
            torch.cuda.empty_cache()
            gc.collect()
            
            logger.info("Successfully loaded embedding layer and lm_head to GPU")
        except Exception as e:
            logger.error(f"Error loading embedding layer: {str(e)}")
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