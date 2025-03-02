import torch
import zmq
from typing import Dict, Any
import logging
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, GenerationConfig, TextStreamer
import time
from ktransformers.optimize.optimize import optimize_and_load_gguf
from ktransformers.models.modeling_deepseek import DeepseekV2ForCausalLM
from ktransformers.models.modeling_qwen2_moe import Qwen2MoeForCausalLM
from ktransformers.models.modeling_deepseek_v3 import DeepseekV3ForCausalLM
from ktransformers.models.modeling_llama import LlamaForCausalLM
from ktransformers.models.modeling_mixtral import MixtralForCausalLM
import multiprocessing as mp
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

custom_models = {
    "DeepseekV2ForCausalLM": DeepseekV2ForCausalLM,
    "DeepseekV3ForCausalLM": DeepseekV3ForCausalLM,
    "Qwen2MoeForCausalLM": Qwen2MoeForCausalLM,
    "LlamaForCausalLM": LlamaForCausalLM,
    "MixtralForCausalLM": MixtralForCausalLM,
}

ktransformer_rules_dir = "/archive/liugu/ds/ktransformers/ktransformers/optimize/optimize_rules/"
default_optimize_rules = {
    "DeepseekV2ForCausalLM": ktransformer_rules_dir + "DeepSeek-V2-Chat.yaml",
    "DeepseekV3ForCausalLM": ktransformer_rules_dir + "DeepSeek-V3-Chat.yaml",
    "Qwen2MoeForCausalLM": ktransformer_rules_dir + "Qwen2-57B-A14B-Instruct.yaml",
    "LlamaForCausalLM": ktransformer_rules_dir + "Internlm2_5-7b-Chat-1m.yaml",
    "MixtralForCausalLM": ktransformer_rules_dir + "Mixtral.yaml",
}

class ModelWorker:
    def __init__(self, address: str, worker_id: int):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        host, port = address.split(':')
        # 每个worker使用不同的端口
        port = int(port) + worker_id
        self.socket.bind(f"tcp://0.0.0.0:{port}")
        self.model = None
        self.tokenizer = None
        self.worker_id = worker_id
        logger.info(f"Worker {worker_id} started at {host}:{port}")

    def load_model(self, model_path: str, gguf_path: str):
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        logger.info("Loading model config...")
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        torch.set_default_dtype(config.torch_dtype)
        
        logger.info("Loading model...")
        if config.architectures[0] in custom_models:
            logger.info("Using custom modeling_xxx.py")
            if "Qwen2Moe" in config.architectures[0]:
                config._attn_implementation = "flash_attention_2"
            if "Llama" in config.architectures[0]:
                config._attn_implementation = "eager"
            if "Mixtral" in config.architectures[0]:
                config._attn_implementation = "flash_attention_2"
            
            self.model = custom_models[config.architectures[0]](config)
        else:
            self.model = AutoModelForCausalLM.from_config(
                config, trust_remote_code=True, attn_implementation="flash_attention_2"
            )
        
        # 使用 ktransformers 的优化
        if config.architectures[0] in default_optimize_rules:
            logger.info(f"Using default_optimize_rule for {config.architectures[0]}")
            optimize_config_path = default_optimize_rules[config.architectures[0]]
        else:
            optimize_config_path = input("Please input optimize rules path: ")
        
        optimize_and_load_gguf(self.model, optimize_config_path, gguf_path, config)
        self.model.eval()
        logger.info("Model loaded successfully")

    def generate(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            logger.info("Received generation request")
            
            input_ids = torch.tensor(input_data['input_ids'], device='cuda')
            attention_mask = torch.tensor(input_data['attention_mask'], device='cuda')
            
            max_new_tokens = input_data.get('max_new_tokens', 50)
            temperature = input_data.get('temperature', 0.6)
            top_p = input_data.get('top_p', 0.9)
            
            logger.info("Starting generation...")
            start_time = time.time()
            
            with torch.no_grad(), torch.amp.autocast('cuda'):
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=input_data.get('do_sample', True)
                )
            
            end_time = time.time()
            generation_time = end_time - start_time
            
            outputs_cpu = outputs.cpu()
            return {
                'output_ids': outputs_cpu.numpy().tolist(),
                'status': 'success',
                'time': generation_time
            }
            
        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }

    def run(self):
        logger.info(f"Worker {self.worker_id} ready to receive requests")
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
                    logger.info(f"Shutting down worker {self.worker_id}")
                    self.socket.send_pyobj({'status': 'shutdown'})
                    break
                
            except Exception as e:
                logger.error(f"Error in worker {self.worker_id}: {str(e)}")
                self.socket.send_pyobj({'status': 'error', 'error': str(e)})

def run_worker(address: str, worker_id: int):
    # 设置进程亲和性
    os.sched_setaffinity(0, {worker_id})
    worker = ModelWorker(address, worker_id)
    worker.run()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker_address', type=str, required=True)
    parser.add_argument('--num_workers', type=int, default=10)
    args = parser.parse_args()
    
    # 创建多个worker进程
    processes = []
    for i in range(args.num_workers):
        p = mp.Process(target=run_worker, args=(args.worker_address, i))
        p.start()
        processes.append(p)
        logger.info(f"Started worker process {i}")
    
    # 等待所有进程完成
    for p in processes:
        p.join()

if __name__ == '__main__':
    main() 