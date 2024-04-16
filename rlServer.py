# -*- coding: utf-8 -*-

from __future__ import division

import subprocess
from concurrent import futures
import gc, grpc
from os import path
import project_root
from Learner import Learner
import indigo_pb2
import indigo_pb2_grpc
import argparse
from helpers.helpers import apply_op
import threading

# 创建一个锁对象
lock = threading.Lock()

# trainer.load_models(2800,s=1,v=15)
def format_actions(action_list):
    """ Returns the action list, initially a list with elements "[op][val]"
    like /2.0, -3.0, +1.0, formatted as a dictionary.

    The dictionary keys are the unique indices (to retrieve the action) and
    the values are lists ['op', val], such as ['+', '2.0'].
    """
    return {idx: [action[0], float(action[1:])]
                  for idx, action in enumerate(action_list)}
class RLmethods(indigo_pb2_grpc.acerServiceServicer):

    def __init__(self):
        self.sample_action = None
        self.learner = None
        self.load_model()

        self.delay = 0.0
        self.delivery_rate = 0.0
        self.send_rate = 0.0
        self.cwnd = 0.0

        self.client_num = 0
        self.client_states = {}

        self.action_mapping = format_actions(["/2.0", "-10.0", "+0.0", "+10.0", "*2.0"])
        self.action_cnt = len(self.action_mapping)

        self.phi = 0.1 # 自己状态所占的比例

    def load_model(self):
        model_path = path.join(project_root.DIR, 'a3c', 'logs', 'model')

        self.learner = Learner(
            state_dim=4,
            action_cnt=5,
            restore_vars=model_path)

        self.sample_action = self.learner.sample_action

    def overly(self, cur_state, avg_state):
        new_state = [0,0,0,0]
        new_state[0] = cur_state[0] * self.phi + avg_state[0] * (1.0 - self.phi)
        new_state[1] = cur_state[1] * self.phi + avg_state[1] * (1.0 - self.phi)
        new_state[2] = cur_state[2] * self.phi + avg_state[2] * (1.0 - self.phi)
        new_state[3] = cur_state[3] * self.phi + avg_state[3] * (1.0 - self.phi)
        return new_state

    def GetAvgState(self):
        return [self.delay / self.client_num,
                self.delivery_rate / self.client_num,
                self.send_rate / self.client_num,
                self.cwnd / self.client_num]

    def GetExplorationAction(self, state, context):
        with lock:
            self.update_states(state)
            port = state.port

            cur_state = [state.delay, state.delivery_rate, state.send_rate, state.cwnd]
            input_state = self.overly(cur_state, self.GetAvgState())

            action = self.sample_action(input_state)
            op, val = self.action_mapping[action]

            target_cwnd = apply_op(op, input_state[3], val)

            self.cwnd += target_cwnd - self.client_states[port][3]
            self.client_states[port][3] = target_cwnd

            print "target_cwnd: " + str(target_cwnd)
            return indigo_pb2.Action(action=target_cwnd)


    def update_states(self, state):
        port = state.port
        if port not in self.client_states:
            self.client_num += 1

        else:
            pre_state = self.client_states[port]
            self.delay -= pre_state[0]
            self.delivery_rate -= pre_state[1]
            self.send_rate -= pre_state[2]
            self.cwnd -= pre_state[3]

        self.client_states[port] = [state.delay, state.delivery_rate, state.send_rate, state.cwnd]
        self.delay += state.delay
        self.delivery_rate += state.delivery_rate
        self.send_rate += state.send_rate
        self.cwnd += state.cwnd

    def UpdateMetric(self, state, context):

        with lock:
            self.update_states(state)
            cur_state = [state.delay, state.delivery_rate, state.send_rate, state.cwnd]
            input_state = self.overly(cur_state, self.GetAvgState())
            print input_state
            print cur_state
            return indigo_pb2.State(delay=input_state[0], delivery_rate=input_state[1], send_rate=input_state[2],
                                    cwnd=input_state[3], port=state.port)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-ip', type=str, default='0.0.0.0', help='rpc listening ip')
    parser.add_argument('-port', type=str, default='50053', help='rpc listening port')

    args = parser.parse_args()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    indigo_pb2_grpc.add_acerServiceServicer_to_server(RLmethods(), server)
    server.add_insecure_port(args.ip + ':' + str(args.port))
    server.start()
    print("rpc serve on %s:%s" % (args.ip, args.port))
    server.wait_for_termination()


if __name__ == '__main__':
    main()