import zmq
import torch
from typing import Dict, Any, Optional, Tuple

class KVCacheClient:
    def __init__(self, server_address: str):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{server_address}")
    
    def store(self, batch_id: str, layer_id: int, k: torch.Tensor, v: torch.Tensor) -> bool:
        message = {
            'command': 'store',
            'batch_id': batch_id,
            'layer_id': layer_id,
            'k': k,
            'v': v
        }
        self.socket.send_pyobj(message)
        response = self.socket.recv_pyobj()
        return response['status'] == 'success'
    
    def retrieve(self, batch_id: str, layer_id: int, device: str) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        message = {
            'command': 'retrieve',
            'batch_id': batch_id,
            'layer_id': layer_id,
            'device': device
        }
        self.socket.send_pyobj(message)
        response = self.socket.recv_pyobj()
        
        if response['status'] == 'success':
            return response['k'], response['v']
        return None
    
    def clear(self, batch_id: str) -> bool:
        message = {
            'command': 'clear',
            'batch_id': batch_id
        }
        self.socket.send_pyobj(message)
        response = self.socket.recv_pyobj()
        return response['status'] == 'success'
    
    def shutdown(self):
        self.socket.send_pyobj({'command': 'shutdown'})
        self.socket.recv_pyobj()
        self.socket.close()
        self.context.term() 