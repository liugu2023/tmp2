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
        torch.set_num_threads(10)
        
        self.model = None
        self.tokenizer = None
        
        # 获取可用的 GPU
        self.num_gpus = torch.cuda.device_count()
        logger.info(f"Found {self.num_gpus} GPUs")
        
        logger.info(f"Worker started at {address}, listening on all interfaces")
        logger.info(f"CPU threads set to: {torch.get_num_threads()}")

    def load_model(self, model_path: str, gguf_path: str):
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        logger.info("Loading model config...")
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        
        # 优化模型配置
        config.use_cache = True
        config.pretraining_tp = 1
        config.torch_dtype = torch.float16
        torch.set_default_dtype(torch.float16)
        
        # 创建设备映射
        if self.num_gpus > 1:
            # 手动创建设备映射，确保使用所有 GPU
            num_layers = config.num_hidden_layers
            layers_per_gpu = num_layers // self.num_gpus
            
            device_map = {
                'model.embed_tokens': 0,
                'model.norm': self.num_gpus - 1,
                'lm_head': self.num_gpus - 1,
            }
            
            # 分配 transformer 层到不同 GPU
            for i in range(num_layers):
                gpu_id = min(i // layers_per_gpu, self.num_gpus - 1)
                device_map[f'model.layers.{i}'] = gpu_id
                
            max_memory = {i: "40GiB" for i in range(self.num_gpus)}
            logger.info(f"Using custom device map across {self.num_gpus} GPUs")
            logger.info(f"Device map: {device_map}")
        else:
            device_map = "auto"
            max_memory = None
        
        logger.info("Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            trust_remote_code=True,
            device_map=device_map,
            max_memory=max_memory,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        self.model.eval()
        
        # 优化生成配置
        try:
            self.model.generation_config = GenerationConfig.from_pretrained(model_path)
        except Exception as e:
            logger.warning(f"Generation config can't auto create, making default. Message: {e}")
            self.model.generation_config = GenerationConfig(
                temperature=0.6,
                top_p=0.9,
                do_sample=True,
                num_beams=1,
                early_stopping=True,
                use_cache=True
            )
        if self.model.generation_config.pad_token_id is None:
            self.model.generation_config.pad_token_id = self.model.generation_config.eos_token_id
            
        logger.info("Model loaded successfully")
        if self.num_gpus > 1:
            logger.info("Model distributed across multiple GPUs")
            for name, param in self.model.named_parameters():
                logger.info(f"Parameter {name} on device: {param.device}")

    def generate(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            logger.info("Received generation request")
            logger.info(f"Input sequence length: {len(input_data['input_ids'])}")
            
            # 对于多 GPU，我们使用第一个 GPU 作为输入设备
            input_device = "cuda:0"
            input_ids = torch.tensor(input_data['input_ids'], device=input_device)
            attention_mask = torch.tensor(input_data['attention_mask'], device=input_device)
            
            # 优化生成参数
            generation_config = {
                'max_new_tokens': input_data.get('max_new_tokens', 512),
                'temperature': input_data.get('temperature', 0.6),
                'top_p': input_data.get('top_p', 0.9),
                'do_sample': input_data.get('do_sample', True),
                # 添加以下参数来加速生成
                'num_beams': 1,  # 禁用 beam search
                'early_stopping': True,
                'repetition_penalty': 1.0,  # 禁用重复惩罚
                'length_penalty': 1.0,  # 中性长度惩罚
                'use_cache': True,  # 启用 KV 缓存
                'pad_token_id': self.model.generation_config.pad_token_id,
                'eos_token_id': self.model.generation_config.eos_token_id,
            }
            
            logger.info(f"Generation parameters: {generation_config}")
            
            logger.info("Starting text generation...")
            with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):  # 使用 FP16
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **generation_config
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