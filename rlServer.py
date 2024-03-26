from __future__ import division
from concurrent import futures
import gc, grpc
from os import path
import project_root
from Learner import Learner
import indigo_pb2
import indigo_pb2_grpc

# trainer.load_models(2800,s=1,v=15)

class RLmethods(indigo_pb2_grpc.acerServiceServicer):

    def __init__(self):
        self.sample_action = None
        self.learner = None
        self.load_model()

        self.sender_num = 10
        self.delay = 0.0
        self.delivery_rate = 0.0
        self.send_rate = 0.0
        self.cwnd = 0.0

        self.phi = 0.1

    def load_model(self):
        model_path = path.join(project_root.DIR, 'a3c', 'logs', 'model')

        self.learner = Learner(
            state_dim=4,
            action_cnt=5,
            restore_vars=model_path)

        self.sample_action = self.learner.sample_action

    def overly(self, cur_state):
        new_state = [0,0,0,0]
        avg_state = [self.delay, self.delivery_rate, self.send_rate, self.cwnd]
        new_state[0] = cur_state[0] * self.phi + avg_state[0] * (1.0 - self.phi)
        new_state[1] = cur_state[1] * self.phi + avg_state[1] * (1.0 - self.phi)
        new_state[2] = cur_state[2] * self.phi + avg_state[2] * (1.0 - self.phi)
        new_state[3] = cur_state[3] * self.phi + avg_state[3] * (1.0 - self.phi)
        return new_state

    def GetExplorationAction(self, state, context):
        cur_state = [state.delay, state.delivery_rate, state.send_rate, state.cwnd]
        input_state = self.overly(cur_state)
        action = self.sample_action(input_state)
        return indigo_pb2.Action(action=action)

    def UpdateMetric(self, state, context):
        self.delay = ((self.sender_num - 1) * self.delay + state[0]) / self.sender_num
        self.delivery_rate = ((self.sender_num - 1) * self.delivery_rate + state[0]) / self.sender_num
        self.send_rate = ((self.sender_num - 1) * self.send_rate + state[0]) / self.sender_num
        self.cwnd = ((self.sender_num - 1) * self.cwnd + state[0]) / self.sender_num

        return indigo_pb2.Empty()

if __name__ == '__main__':
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    indigo_pb2_grpc.add_acerServiceServicer_to_server(RLmethods(), server)
    server.add_insecure_port('[::]:50053')
    server.start()
    print("rpc serve in port 50053")
    server.wait_for_termination()
