import torch
import zmq
import pickle
from typing import Dict, Any
import logging
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, GenerationConfig
import time
from transformers import TextStreamer

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
        
        logger.info("Model loaded successfully on CPU")

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
                    
                    # 使用自定义前向传播，让模型权重保持在CPU上
                    def custom_forward(input_ids, attention_mask):
                        # 逐层处理，每层的权重从CPU读取，计算在GPU上进行
                        hidden_states = self.model.model.embed_tokens(input_ids)
                        
                        for layer in self.model.model.layers:
                            # 将当前层的输入移到GPU
                            hidden_states = hidden_states.cuda()
                            # 层的计算在GPU上进行，但权重从CPU读取
                            hidden_states = layer(hidden_states, attention_mask)[0]
                            # 将结果移回CPU，释放GPU内存
                            hidden_states = hidden_states.cpu()
                        
                        # 最后的处理
                        hidden_states = hidden_states.cuda()
                        hidden_states = self.model.model.norm(hidden_states)
                        logits = self.model.lm_head(hidden_states)
                        return logits
                    
                    # 使用自定义生成函数
                    outputs = []
                    current_input_ids = input_ids
                    current_attention_mask = attention_mask
                    
                    for _ in range(max_new_tokens):
                        # 获取下一个token
                        logits = custom_forward(current_input_ids, current_attention_mask)
                        next_token_logits = logits[:, -1, :]
                        
                        # 采样下一个token
                        if input_data.get('do_sample', True):
                            probs = torch.nn.functional.softmax(next_token_logits / temperature, dim=-1)
                            next_token = torch.multinomial(probs, num_samples=1)
                        else:
                            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                        
                        # 添加到输出序列
                        current_input_ids = torch.cat([current_input_ids, next_token], dim=1)
                        current_attention_mask = torch.cat([current_attention_mask, torch.ones_like(next_token)], dim=1)
                        
                        # 检查是否生成了结束标记
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