import torch
import zmq
from typing import Dict, Any
import logging
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, GenerationConfig, TextStreamer
import time

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
            
            # 创建输入张量在GPU上
            logger.info("Creating input tensors on GPU...")
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
            
            # 使用 CUDA 内存管理
            with torch.cuda.stream(torch.cuda.Stream()):
                with torch.no_grad(), torch.amp.autocast('cuda'):
                    logger.info("Generating text...")
                    
                    # 修改模型的前向传播方法
                    original_forward = self.model.forward
                    
                    def gpu_forward(*args, **kwargs):
                        # 输入已经在GPU上，不需要移动
                        gpu_args = args
                        gpu_kwargs = kwargs
                        
                        # 对每个模块的权重进行GPU计算
                        for name, module in self.model.named_modules():
                            if hasattr(module, 'weight') and module.weight is not None:
                                # 临时将权重移到GPU
                                with torch.cuda.device('cuda'):
                                    weight = module.weight.cuda()
                                    if hasattr(module, 'bias') and module.bias is not None:
                                        bias = module.bias.cuda()
                                    # 执行计算
                                    if isinstance(module, torch.nn.Linear):
                                        output = torch.nn.functional.linear(gpu_args[0], weight, bias if module.bias is not None else None)
                                    elif isinstance(module, torch.nn.LayerNorm):
                                        output = torch.nn.functional.layer_norm(gpu_args[0], module.normalized_shape, weight, bias if module.bias is not None else None)
                                    # 结果已经在GPU上
                                    gpu_args = (output,) + gpu_args[1:]
                        
                        # 返回最终结果（在GPU上）
                        return gpu_args[0]
                    
                    # 替换模型的前向传播方法
                    self.model.forward = gpu_forward
                    
                    try:
                        # 生成文本
                        outputs = self.model.generate(
                            input_ids=input_ids,  # 已经在GPU上
                            attention_mask=attention_mask,  # 已经在GPU上
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            do_sample=input_data.get('do_sample', True),
                            streamer=TextStreamer(self.tokenizer, skip_prompt=True)
                        )
                    finally:
                        # 恢复原始的前向传播方法
                        self.model.forward = original_forward
                
                # 将输出移到CPU进行后处理
                outputs = outputs.cpu()
            
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