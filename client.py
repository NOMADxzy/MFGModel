# client.py
import grpc

import indigo_pb2
import indigo_pb2_grpc

def run():
    channel = grpc.insecure_channel('localhost:50053')
    stub = indigo_pb2_grpc.acerServiceStub(channel)
    response = stub.GetExplorationAction(indigo_pb2.State(delay=20, delivery_rate=10, send_rate=10, cwnd=3, port=1))
    # response = stub.GetExplorationAction(indigo_pb2.State(delay=10, delivery_rate=10, send_rate=10, cwnd=10, port=8081))
    response = stub.UpdateMetric(indigo_pb2.State(delay=20, delivery_rate=10, send_rate=10, cwnd=3, port=1))
    print("Greeter client received: " + str(response.action))

if __name__ == '__main__':
    run()
