import torch
import zmq
import pickle
from typing import Dict, Any
import logging
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, GenerationConfig
import time
from transformers import TextStreamer
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    def __init__(self, address: str):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        host, port = address.split(':')
        self.socket.bind(f"tcp://0.0.0.0:{port}")
        self.socket.setsockopt(zmq.RCVHWM, 0)
        self.socket.setsockopt(zmq.SNDHWM, 0)
        
        self.model = None
        self.tokenizer = None
        # 添加缓存字典
        self.layer_cache = {}
        self.embedding_cache = None
        self.norm_cache = None
        self.lm_head_cache = None
        logger.info(f"Worker started at {address}, listening on all interfaces")

    def load_model(self, model_path: str, gguf_path: str):
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        logger.info("Loading model config...")
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        torch.set_default_dtype(config.torch_dtype)
        
        logger.info("Loading model on CPU...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            trust_remote_code=True,
            device_map='cpu',  # 在CPU上加载
            torch_dtype=torch.float16,  # 使用半精度
            low_cpu_mem_usage=True,     # 降低CPU内存使用
        )
        self.model.eval()
        
        # 预处理并缓存模型的各个部分
        logger.info("Preprocessing and caching model components...")
        self.preprocess_model()
        
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
        
        logger.info("Model loaded and cached successfully")

    def preprocess_model(self):
        """预处理模型并缓存各个组件"""
        logger.info("Caching model components...")
        
        # 缓存嵌入层
        self.embedding_cache = {
            'weight': self.model.model.embed_tokens.weight.cpu(),
            'device': self.model.model.embed_tokens.weight.device
        }
        
        # 缓存每个transformer层
        for i, layer in enumerate(self.model.model.layers):
            self.layer_cache[i] = {
                'self_attn': {
                    'q_proj': {'weight': layer.self_attn.q_proj.weight.cpu(), 'bias': layer.self_attn.q_proj.bias.cpu() if layer.self_attn.q_proj.bias is not None else None},
                    'k_proj': {'weight': layer.self_attn.k_proj.weight.cpu(), 'bias': layer.self_attn.k_proj.bias.cpu() if layer.self_attn.k_proj.bias is not None else None},
                    'v_proj': {'weight': layer.self_attn.v_proj.weight.cpu(), 'bias': layer.self_attn.v_proj.bias.cpu() if layer.self_attn.v_proj.bias is not None else None},
                    'o_proj': {'weight': layer.self_attn.o_proj.weight.cpu(), 'bias': layer.self_attn.o_proj.bias.cpu() if layer.self_attn.o_proj.bias is not None else None},
                },
                'mlp': {
                    'gate_proj': {'weight': layer.mlp.gate_proj.weight.cpu()},
                    'up_proj': {'weight': layer.mlp.up_proj.weight.cpu()},
                    'down_proj': {'weight': layer.mlp.down_proj.weight.cpu()},
                },
                'input_layernorm': {'weight': layer.input_layernorm.weight.cpu()},
                'post_attention_layernorm': {'weight': layer.post_attention_layernorm.weight.cpu()}
            }
        
        # 缓存最终的norm层
        self.norm_cache = {
            'weight': self.model.model.norm.weight.cpu()
        }
        
        # 缓存语言模型头
        self.lm_head_cache = {
            'weight': self.model.lm_head.weight.cpu()
        }
        
        logger.info("Model components cached successfully")

    def compute_attention(self, q, k, v, attention_mask):
        """计算注意力"""
        # 获取关键维度
        batch_size, seq_len, num_heads, head_dim = q.size()
        
        # 重塑 q, k, v 为多头注意力格式
        q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)  # [batch, heads, seq_len, head_dim]
        k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        
        # 计算注意力分数
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        
        # 处理注意力掩码
        # 确保掩码的形状正确
        if attention_mask is not None:
            # 扩展掩码维度以匹配注意力分数
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
            attention_mask = attention_mask.expand(-1, num_heads, -1, -1)  # [batch, heads, 1, seq_len]
            attention_scores = attention_scores + attention_mask
        
        # 应用 softmax
        attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
        
        # 计算输出
        context = torch.matmul(attention_probs, v)  # [batch, heads, seq_len, head_dim]
        
        # 重塑回原始维度
        context = context.transpose(1, 2).contiguous()  # [batch, seq_len, heads, head_dim]
        context = context.view(batch_size, seq_len, -1)  # [batch, seq_len, hidden_size]
        
        return context

    def custom_forward_with_cache(self, input_ids, attention_mask):
        """使用缓存的模型组件进行前向传播"""
        # 使用缓存的嵌入层
        hidden_states = torch.nn.functional.embedding(
            input_ids.cuda(),
            self.embedding_cache['weight'].cuda()
        )
        
        # 获取模型配置
        config = self.model.config
        head_dim = config.hidden_size // config.num_attention_heads
        
        # 处理注意力掩码
        attention_mask = (attention_mask < 0).float() * -10000.0
        
        # 逐层处理
        for i in range(len(self.layer_cache)):
            layer_cache = self.layer_cache[i]
            
            # 自注意力处理
            qkv_states = []
            for proj in ['q_proj', 'k_proj', 'v_proj']:
                proj_cache = layer_cache['self_attn'][proj]
                states = torch.nn.functional.linear(
                    hidden_states.cuda(),
                    proj_cache['weight'].cuda(),
                    proj_cache['bias'].cuda() if proj_cache['bias'] is not None else None
                )
                # 重塑为多头格式
                states = states.view(states.size(0), -1, config.num_attention_heads, head_dim)
                qkv_states.append(states)
            
            # 计算注意力
            q, k, v = qkv_states
            attention_output = self.compute_attention(q, k, v, attention_mask)
            
            # 输出投影
            attention_output = torch.nn.functional.linear(
                attention_output,
                layer_cache['self_attn']['o_proj']['weight'].cuda(),
                layer_cache['self_attn']['o_proj']['bias'].cuda() if layer_cache['self_attn']['o_proj']['bias'] is not None else None
            )
            
            # 残差连接和层标准化
            hidden_states = attention_output + hidden_states
            hidden_states = self.apply_layernorm(hidden_states, layer_cache['input_layernorm']['weight'].cuda())
            
            # MLP处理
            mlp_output = self.compute_mlp(hidden_states, layer_cache['mlp'])
            hidden_states = mlp_output + hidden_states
            hidden_states = self.apply_layernorm(hidden_states, layer_cache['post_attention_layernorm']['weight'].cuda())
            
            # 将中间结果移回CPU以节省GPU内存
            hidden_states = hidden_states.cpu()
        
        # 最终的norm和语言模型头
        hidden_states = self.apply_layernorm(hidden_states.cuda(), self.norm_cache['weight'].cuda())
        logits = torch.nn.functional.linear(hidden_states, self.lm_head_cache['weight'].cuda())
        
        return logits

    def compute_mlp(self, hidden_states, mlp_cache):
        """计算MLP层"""
        gate_output = torch.nn.functional.linear(hidden_states, mlp_cache['gate_proj']['weight'].cuda())
        up_output = torch.nn.functional.linear(hidden_states, mlp_cache['up_proj']['weight'].cuda())
        activated = torch.nn.functional.silu(gate_output) * up_output
        return torch.nn.functional.linear(activated, mlp_cache['down_proj']['weight'].cuda())

    def apply_layernorm(self, hidden_states, weight):
        """应用层标准化"""
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + 1e-5)
        return hidden_states * weight

    def generate(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            logger.info("Received generation request")
            logger.info(f"Input sequence length: {len(input_data['input_ids'])}")
            
            # 创建输入张量在GPU上
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
            
            # 设置模型以使用CPU内存但在GPU上计算
            self.model.to('cpu')  # 确保模型在CPU内存中
            torch.cuda.empty_cache()  # 清理GPU内存
            
            # 启用CPU-GPU数据流水线
            with torch.cuda.stream(torch.cuda.Stream()):
                with torch.no_grad(), torch.amp.autocast('cuda'):
                    logger.info("Initializing generation process...")
                    # 添加详细的初始化信息
                    logger.info("- Checking model state...")
                    logger.info(f"- Model device: {self.model.device}")
                    logger.info(f"- Input shape: {input_ids.shape}")
                    logger.info(f"- Current GPU memory allocated: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
                    
                    # KV cache 信息
                    num_layers = len(self.model.model.layers)
                    hidden_size = self.model.config.hidden_size
                    logger.info(f"- Model architecture:")
                    logger.info(f"  - Number of layers: {num_layers}")
                    logger.info(f"  - Hidden size: {hidden_size}")
                    
                    # 估算 KV cache 大小
                    seq_len = input_ids.shape[1] + max_new_tokens
                    kv_cache_size = 2 * num_layers * seq_len * hidden_size * input_ids.shape[0] * 2 / (1024**3)  # in GB
                    logger.info(f"- Estimated KV cache size: {kv_cache_size:.2f}GB")
                    
                    # 设置生成配置
                    logger.info("  - Generation started with following settings:")
                    logger.info(f"    - Input context length: {len(input_ids[0])} tokens")
                    logger.info(f"    - Maximum new tokens: {max_new_tokens}")
                    logger.info(f"    - Temperature: {temperature}")
                    logger.info(f"    - Top-p sampling: {top_p}")
                    
                    # 使用带缓存的自定义前向传播
                    outputs = []
                    current_input_ids = input_ids
                    current_attention_mask = attention_mask
                    
                    for _ in range(max_new_tokens):
                        logits = self.custom_forward_with_cache(current_input_ids, current_attention_mask)
                        next_token_logits = logits[:, -1, :]
                        
                        if input_data.get('do_sample', True):
                            probs = torch.nn.functional.softmax(next_token_logits / temperature, dim=-1)
                            next_token = torch.multinomial(probs, num_samples=1)
                        else:
                            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                        
                        current_input_ids = torch.cat([current_input_ids, next_token], dim=1)
                        current_attention_mask = torch.cat([current_attention_mask, torch.ones_like(next_token)], dim=1)
                        
                        if next_token[0].item() == self.model.generation_config.eos_token_id:
                            break
                    
                    outputs = current_input_ids
            
            # 清理GPU内存
            torch.cuda.empty_cache()
            
            # 移动输出到CPU
            logger.info("Moving outputs to CPU...")
            outputs_cpu = outputs.cpu()
            
            # 记录GPU内存使用情况
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"GPU Memory: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")
            
            logger.info("Preparing response...")
            return {
                'output_ids': outputs_cpu.numpy().tolist(),
                'status': 'success'
            }
        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            torch.cuda.empty_cache()
            return {
                'status': 'error',
                'error': str(e)
            }

    def run(self):
        logger.info("Worker ready to receive requests")
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
                    logger.info("Shutting down worker")
                    self.socket.send_pyobj({'status': 'shutdown'})
                    break
                
            except Exception as e:
                logger.error(f"Error processing message: {str(e)}")
                self.socket.send_pyobj({'status': 'error', 'error': str(e)})

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker_address', type=str, required=True)
    args = parser.parse_args()
    
    worker = ModelWorker(args.worker_address)
    worker.run()

if __name__ == '__main__':
    main() 