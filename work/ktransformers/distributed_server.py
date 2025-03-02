import os
import argparse
from concurrent import futures
import grpc
import chat_pb2
import chat_pb2_grpc
from worker import DistributedWorker

class ChatService(chat_pb2_grpc.ChatServiceServicer):
    def __init__(self, worker):
        self.worker = worker
        
    def Generate(self, request, context):
        try:
            response = self.worker.generate_response(
                prompt=request.prompt,
                max_new_tokens=request.max_new_tokens,
                mode=request.mode,
                force_think=request.force_think
            )
            return chat_pb2.GenerateResponse(text=response)
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return chat_pb2.GenerateResponse()

def serve(args):
    worker = DistributedWorker(
        host=args.host,
        device_type=args.device_type,
        device_id=args.device_id,
        model_path=args.model_path,
        gguf_path=args.gguf_path,
        optimize_config_path=args.optimize_config_path
    )
    
    worker.initialize()
    
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    chat_pb2_grpc.add_ChatServiceServicer_to_server(ChatService(worker), server)
    server.add_insecure_port(f'{args.host}:{args.port}')
    server.start()
    print(f"服务器启动在 {args.host}:{args.port}")
    server.wait_for_termination()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='启动分布式聊天服务器')
    parser.add_argument('--host', type=str, default='[::]',
                       help='服务器主机地址')
    parser.add_argument('--port', type=int, default=50051,
                       help='服务器端口')
    parser.add_argument('--model_path', type=str, required=True,
                       help='模型路径')
    parser.add_argument('--gguf_path', type=str, required=True,
                       help='GGUF模型路径')
    parser.add_argument('--optimize_config_path', type=str, default=None,
                       help='优化配置路径')
    parser.add_argument('--device_type', type=str, default='gpu',
                       choices=['cpu', 'gpu'], help='运行设备类型')
    parser.add_argument('--device_id', type=int, default=0,
                       help='GPU设备ID')
    
    args = parser.parse_args()
    serve(args) 