# client.py
import grpc

import indigo_pb2
import indigo_pb2_grpc

def run():
    channel = grpc.insecure_channel('localhost:50054')
    stub = indigo_pb2_grpc.acerServiceStub(channel)
    response = stub.GetExplorationAction(indigo_pb2.State(delay=20, delivery_rate=20, send_rate=20, cwnd=20, port=8080))
    # response = stub.GetExplorationAction(indigo_pb2.State(delay=10, delivery_rate=10, send_rate=10, cwnd=10, port=8081))
    print("Greeter client received: " + str(response.action))

if __name__ == '__main__':
    run()
