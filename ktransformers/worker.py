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
from queue import Empty
from multiprocessing import Queue

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
    def __init__(self, worker_id: int, task_queue: Queue, result_queue: Queue):
        self.worker_id = worker_id
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.model = None
        self.tokenizer = None
        logger.info(f"Worker {worker_id} initialized")

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
        logger.info(f"Worker {self.worker_id} ready to process tasks")
        while True:
            try:
                task = self.task_queue.get()
                if task is None:  # 停止信号
                    break
                
                command = task.get('command')
                task_id = task.get('task_id')
                
                if command == 'load_model':
                    self.load_model(task['model_path'], task['gguf_path'])
                    self.result_queue.put({
                        'task_id': task_id,
                        'status': 'success'
                    })
                
                elif command == 'generate':
                    result = self.generate(task['data'])
                    result['task_id'] = task_id
                    self.result_queue.put(result)
                
            except Exception as e:
                logger.error(f"Error in worker {self.worker_id}: {str(e)}")
                self.result_queue.put({
                    'task_id': task.get('task_id'),
                    'status': 'error',
                    'error': str(e)
                })

def run_worker(worker_id: int, task_queue: Queue, result_queue: Queue):
    # 设置进程亲和性
    os.sched_setaffinity(0, {worker_id})
    worker = ModelWorker(worker_id, task_queue, result_queue)
    worker.run()

class ModelServer:
    def __init__(self, address: str, num_workers: int):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://{address}")
        self.num_workers = num_workers
        self.task_queue = Queue()
        self.result_queue = Queue()
        self.task_id = 0
        self.workers = []
        
        # 启动工作进程
        for i in range(num_workers):
            p = mp.Process(target=run_worker, args=(i, self.task_queue, self.result_queue))
            p.start()
            self.workers.append(p)
            logger.info(f"Started worker process {i}")

    def run(self):
        logger.info("Server started, waiting for requests")
        try:
            while True:
                message = self.socket.recv_pyobj()
                self.task_id += 1
                message['task_id'] = self.task_id
                
                # 发送任务给工作进程
                self.task_queue.put(message)
                
                # 等待结果
                result = self.result_queue.get()
                self.socket.send_pyobj(result)
                
                if message.get('command') == 'shutdown':
                    break
                    
        finally:
            # 发送停止信号给所有工作进程
            for _ in range(self.num_workers):
                self.task_queue.put(None)
            
            # 等待所有工作进程结束
            for p in self.workers:
                p.join()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker_address', type=str, required=True)
    parser.add_argument('--num_workers', type=int, default=10)
    args = parser.parse_args()
    
    server = ModelServer(args.worker_address, args.num_workers)
    server.run()

if __name__ == '__main__':
    main() 