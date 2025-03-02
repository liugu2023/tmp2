import torch
import zmq
from typing import Dict, Any
import logging
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, GenerationConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CPUWorker:
    def __init__(self, address: str):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        host, port = address.split(':')
        self.socket.bind(f"tcp://0.0.0.0:{port}")
        self.socket.setsockopt(zmq.RCVHWM, 0)
        self.socket.setsockopt(zmq.SNDHWM, 0)
        
        # 设置 CPU 线程数
        torch.set_num_threads(30)
        
        self.model = None
        self.tokenizer = None
        logger.info(f"CPU Worker started at {address}")
        logger.info(f"CPU threads: {torch.get_num_threads()}")

    def run(self):
        logger.info("CPU Worker ready to receive requests")
        while True:
            try:
                message = self.socket.recv_pyobj()
                command = message.get('command')
                
                if command == 'process_cpu_layers':
                    # 处理 CPU 层的计算
                    result = self.process_cpu_layers(message['data'])
                    self.socket.send_pyobj(result)
                
                elif command == 'shutdown':
                    logger.info("Shutting down CPU worker")
                    self.socket.send_pyobj({'status': 'shutdown'})
                    break
                
            except Exception as e:
                logger.error(f"Error processing message: {str(e)}")
                self.socket.send_pyobj({'status': 'error', 'error': str(e)})

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--address', type=str, required=True)
    args = parser.parse_args()
    
    worker = CPUWorker(args.address)
    worker.run()

if __name__ == '__main__':
    main() 