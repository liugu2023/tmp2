import grpc
import argparse
import chat_pb2
import chat_pb2_grpc

def run_chat(host: str, port: int):
    channel = grpc.insecure_channel(f'{host}:{port}')
    stub = chat_pb2_grpc.ChatServiceStub(channel)
    
    print("开始聊天（输入'quit'退出）：")
    while True:
        content = input("\n用户: ")
        if content.lower() == 'quit':
            break
            
        if content.startswith('"""'):  # 多行输入支持
            content = content[3:] + "\n"
            while True:
                line = input("")
                if line.endswith('"""'):
                    line = line[:-3]
                    if line:
                        content += line + "\n"
                    break
                else:
                    content += line + "\n"
                    
        request = chat_pb2.GenerateRequest(
            prompt=content,
            max_new_tokens=300,
            mode="normal",
            force_think=False
        )
        
        try:
            response = stub.Generate(request)
            print(f"\n助手: {response.text}")
        except grpc.RpcError as e:
            print(f"错误: {e.details()}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='分布式聊天客户端')
    parser.add_argument('--host', type=str, default='localhost',
                       help='服务器主机地址')
    parser.add_argument('--port', type=int, default=50051,
                       help='服务器端口')
    
    args = parser.parse_args()
    run_chat(args.host, args.port) 