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
        
        # 设置 CPU 线程数
        torch.set_num_threads(10)  # 限制 CPU 线程数为 10
        
        self.model = None
        self.tokenizer = None
        logger.info(f"Worker started at {address}, listening on all interfaces")
        logger.info(f"CPU threads set to: {torch.get_num_threads()}")

    def load_model(self, model_path: str, gguf_path: str):
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        logger.info("Loading model config...")
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        torch.set_default_dtype(config.torch_dtype)
        
        logger.info("Loading model...")
        # 设置设备映射，将模型分布到两个 GPU 上
        device_map = {
            'model.embed_tokens': 0,
            'model.layers.0': 0,
            'model.layers.1': 0,
            'model.layers.2': 0,
            'model.layers.3': 0,
            'model.layers.4': 0,
            'model.layers.5': 0,
            'model.layers.6': 0,
            'model.layers.7': 0,
            'model.layers.8': 0,
            'model.layers.9': 0,
            'model.layers.10': 0,
            'model.layers.11': 0,
            'model.layers.12': 0,
            'model.layers.13': 0,
            'model.layers.14': 0,
            'model.layers.15': 1,
            'model.layers.16': 1,
            'model.layers.17': 1,
            'model.layers.18': 1,
            'model.layers.19': 1,
            'model.layers.20': 1,
            'model.layers.21': 1,
            'model.layers.22': 1,
            'model.layers.23': 1,
            'model.layers.24': 1,
            'model.layers.25': 1,
            'model.layers.26': 1,
            'model.layers.27': 1,
            'model.layers.28': 1,
            'model.layers.29': 1,
            'model.layers.30': 1,
            'model.norm': 1,
            'lm_head': 1
        }
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            trust_remote_code=True,
            device_map=device_map,  # 使用自定义设备映射
            torch_dtype=torch.bfloat16  # 使用 bfloat16 进一步优化性能
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
            
            # 将输入放在第一个 GPU 上
            input_ids = torch.tensor(input_data['input_ids'], device='cuda:0')
            attention_mask = torch.tensor(input_data['attention_mask'], device='cuda:0')
            
            logger.info(f"Generation parameters: max_new_tokens={input_data.get('max_new_tokens', 512)}, "
                       f"temperature={input_data.get('temperature', 0.6)}, "
                       f"top_p={input_data.get('top_p', 0.9)}")
            
            logger.info("Starting text generation...")
            with torch.no_grad(), torch.amp.autocast('cuda'):
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=input_data.get('max_new_tokens', 512),
                    temperature=input_data.get('temperature', 0.6),
                    top_p=input_data.get('top_p', 0.9),
                    do_sample=input_data.get('do_sample', True),
                    use_cache=True  # 启用 KV 缓存
                )
            
            logger.info(f"Generation completed. Output sequence length: {len(outputs[0])}")
            
            outputs_cpu = outputs.to('cpu', non_blocking=True)
            torch.cuda.synchronize()
            
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