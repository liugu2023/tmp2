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
    def __init__(self, address: str, cpu_worker_address: str):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        host, port = address.split(':')
        self.socket.bind(f"tcp://0.0.0.0:{port}")
        self.socket.setsockopt(zmq.RCVHWM, 0)
        self.socket.setsockopt(zmq.SNDHWM, 0)
        
        # 连接到 CPU worker
        self.cpu_socket = self.context.socket(zmq.REQ)
        self.cpu_socket.connect(f"tcp://{cpu_worker_address}")
        
        # 设置 GPU 内存限制
        torch.cuda.set_per_process_memory_fraction(0.88)
        
        self.model = None
        self.tokenizer = None
        logger.info(f"GPU Worker started at {address}")
        logger.info(f"Connected to CPU Worker at {cpu_worker_address}")

    def load_model(self, model_path: str, gguf_path: str):
        try:
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            
            logger.info("Loading model config...")
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            torch.set_default_dtype(config.torch_dtype)
            
            logger.info("Loading model...")
            num_layers = config.num_hidden_layers
            
            # 使用默认的设备映射策略进行初始加载
            device_map = "auto"
            max_memory = {0: "14GB", "cpu": "128GB"}
            
            logger.info("Using auto device map for initial loading")
            
            # 加载模型
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                config=config,
                trust_remote_code=True,
                device_map=device_map,
                max_memory=max_memory,
                offload_folder="offload",
                torch_dtype=torch.float16
            )
            self.model.eval()
            
            # 记录初始设备分配
            logger.info("Initial device map:")
            for name, device in self.model.hf_device_map.items():
                logger.info(f"{name}: {device}")
            
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
            
            logger.info("Model loading completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False

    def generate(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            logger.info("Received generation request")
            logger.info(f"Input sequence length: {len(input_data['input_ids'])}")
            
            # 创建输入张量并放在 GPU 上
            input_ids = torch.tensor(input_data['input_ids'], device='cuda')
            attention_mask = torch.tensor(input_data['attention_mask'], device='cuda')
            
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
                    use_cache=True
                )
            
            logger.info(f"Generation completed. Output sequence length: {len(outputs[0])}")
            
            # 异步传输到 CPU
            outputs_cpu = outputs.to('cpu', non_blocking=True)
            torch.cuda.synchronize()
            
            # 清理 CUDA 缓存
            torch.cuda.empty_cache()
            
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
    parser.add_argument('--cpu_worker_address', type=str, required=True)
    args = parser.parse_args()
    
    worker = ModelWorker(args.worker_address, args.cpu_worker_address)
    worker.run()

if __name__ == '__main__':
    main() 