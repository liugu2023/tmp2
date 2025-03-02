import torch
import zmq
import pickle
from typing import Dict, Any
import logging
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, GenerationConfig

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
        
        logger.info("Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            trust_remote_code=True,
            device_map="auto"
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
            
            # 创建输入张量
            logger.info("Moving input tensors to GPU...")
            input_ids = torch.tensor(input_data['input_ids'], device='cuda')
            attention_mask = torch.tensor(input_data['attention_mask'], device='cuda')
            
            # 生成参数日志
            logger.info(f"Generation parameters: max_new_tokens={input_data.get('max_new_tokens', 512)}, "
                       f"temperature={input_data.get('temperature', 0.6)}, "
                       f"top_p={input_data.get('top_p', 0.9)}")
            
            # 生成文本
            logger.info("Starting text generation...")
            with torch.no_grad(), torch.amp.autocast('cuda'):
                # 添加生成过程的详细日志
                logger.info("Initializing generation process...")
                
                def logging_callback(step: int, token_id: int, token_str: str):
                    if step % 10 == 0:  # 每10个token记录一次
                        logger.info(f"Generated {step} tokens...")
                
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=input_data.get('max_new_tokens', 512),
                    temperature=input_data.get('temperature', 0.6),
                    top_p=input_data.get('top_p', 0.9),
                    do_sample=input_data.get('do_sample', True),
                    callback=logging_callback,  # 添加回调函数
                    callback_interval=1  # 每生成一个token调用一次
                )
                
                logger.info(f"Generation completed. Total tokens generated: {len(outputs[0]) - len(input_ids[0])}")
                logger.info(f"Final sequence length: {len(outputs[0])}")
            
            # 移动到CPU
            logger.info("Moving outputs to CPU...")
            outputs_cpu = outputs.to('cpu', non_blocking=True)
            torch.cuda.synchronize()
            
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