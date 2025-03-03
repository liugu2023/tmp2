import torch
import zmq
from typing import Dict, Any
import logging
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, GenerationConfig, TextStreamer
import time
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        
        logger.info("Model loaded successfully")

    def generate(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            logger.info("Received generation request")
            logger.info(f"Input sequence length: {len(input_data['input_ids'])}")
            
            # 创建输入张量在CPU上（与模型保持一致）
            logger.info("Creating input tensors...")
            input_ids = torch.tensor(input_data['input_ids'])
            attention_mask = torch.tensor(input_data['attention_mask'])
            
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
            
            # 使用 CUDA 内存管理
            with torch.cuda.stream(torch.cuda.Stream()):
                with torch.no_grad(), torch.amp.autocast('cuda'):
                    logger.info("Generating text...")
                    
                    # 修改模型的前向传播方法
                    original_forward = self.model.forward
                    
                    def gpu_forward(*args, **kwargs):
                        # 获取输入
                        hidden_states = args[0]
                        
                        # 对每个层进行GPU计算
                        for layer in self.model.model.layers:
                            # 自注意力层
                            qkv_weight = torch.cat([
                                layer.self_attn.q_proj.weight,
                                layer.self_attn.k_proj.weight,
                                layer.self_attn.v_proj.weight
                            ]).cuda()
                            
                            # 计算 Q, K, V
                            hidden_gpu = hidden_states.cuda()
                            qkv = torch.nn.functional.linear(hidden_gpu, qkv_weight)
                            q, k, v = qkv.chunk(3, dim=-1)
                            
                            # 计算注意力
                            attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
                            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
                            attn_output = torch.matmul(attn_weights, v)
                            
                            # 输出投影
                            attn_output = torch.nn.functional.linear(
                                attn_output,
                                layer.self_attn.o_proj.weight.cuda(),
                                layer.self_attn.o_proj.bias.cuda() if layer.self_attn.o_proj.bias is not None else None
                            )
                            
                            # 第一个残差连接
                            hidden_states = (hidden_gpu + attn_output).cpu()
                            
                            # 第一个层标准化
                            hidden_gpu = hidden_states.cuda()
                            hidden_states = torch.nn.functional.layer_norm(
                                hidden_gpu,
                                layer.input_layernorm.normalized_shape,
                                layer.input_layernorm.weight.cuda(),
                                layer.input_layernorm.bias.cuda() if layer.input_layernorm.bias is not None else None
                            ).cpu()
                            
                            # MLP
                            hidden_gpu = hidden_states.cuda()
                            mlp_output = torch.nn.functional.linear(hidden_gpu, layer.mlp.gate_proj.weight.cuda())
                            mlp_output = torch.nn.functional.silu(mlp_output)
                            mlp_output = torch.nn.functional.linear(mlp_output, layer.mlp.down_proj.weight.cuda())
                            
                            # 第二个残差连接
                            hidden_states = (hidden_gpu + mlp_output).cpu()
                            
                            # 第二个层标准化
                            hidden_gpu = hidden_states.cuda()
                            hidden_states = torch.nn.functional.layer_norm(
                                hidden_gpu,
                                layer.post_attention_layernorm.normalized_shape,
                                layer.post_attention_layernorm.weight.cuda(),
                                layer.post_attention_layernorm.bias.cuda() if layer.post_attention_layernorm.bias is not None else None
                            ).cpu()
                        
                        # 最终的层标准化
                        hidden_gpu = hidden_states.cuda()
                        hidden_states = torch.nn.functional.layer_norm(
                            hidden_gpu,
                            self.model.model.norm.normalized_shape,
                            self.model.model.norm.weight.cuda(),
                            self.model.model.norm.bias.cuda() if self.model.model.norm.bias is not None else None
                        ).cpu()
                        
                        # 语言模型头
                        hidden_gpu = hidden_states.cuda()
                        logits = torch.nn.functional.linear(hidden_gpu, self.model.lm_head.weight.cuda())
                        
                        return logits.cpu()
                    
                    # 替换模型的前向传播方法
                    self.model.forward = gpu_forward
                    
                    try:
                        # 生成文本
                        outputs = self.model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            do_sample=input_data.get('do_sample', True),
                            streamer=TextStreamer(self.tokenizer, skip_prompt=True)
                        )
                    finally:
                        # 恢复原始的前向传播方法
                        self.model.forward = original_forward
                
                # 清理GPU内存
                torch.cuda.empty_cache()
                
                # 记录GPU内存使用情况
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                logger.info(f"GPU Memory: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")
                
                logger.info("Preparing response...")
                return {
                    'output_ids': outputs.numpy().tolist(),
                    'status': 'success'
                }
        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            # 清理GPU内存
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