# client.py
import grpc

import indigo_pb2
import indigo_pb2_grpc

def run():
    channel = grpc.insecure_channel('localhost:50053')
    stub = indigo_pb2_grpc.acerServiceStub(channel)
    response = stub.GetExplorationAction(indigo_pb2.State(delay=25, delivery_rate=100, send_rate=100, cwnd=10))
    print("Greeter client received: " + str(response.action))

if __name__ == '__main__':
    run()
