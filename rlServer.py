# -*- coding: utf-8 -*-

from __future__ import division

import os.path
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
from filelock import Timeout, FileLock
import logging

# 全局配置，默认的日志级别设置为 WARNING，
# 这样 INFO 和 DEBUG 级别的日志则不会显示
logging.basicConfig(level=logging.INFO)

# 创建一个锁对象
lock = FileLock("state.lock")


# lock = threading.Lock()

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

        # self.delay = 0.0
        # self.delivery_rate = 0.0
        # self.send_rate = 0.0
        self.cwnd = 0.0

        # self.client_num = 0
        # self.client_states = {}
        self.pre_state = [0, 0, 0, 0]

        self.action_mapping = format_actions(["/2.0", "-10.0", "+0.0", "+10.0", "*2.0"])
        self.action_cnt = len(self.action_mapping)

        self.phi = 0.1  # 自己状态所占的比例

    def load_model(self):
        model_path = path.join(project_root.DIR, 'a3c', 'logs', 'model')

        self.learner = Learner(
            state_dim=4,
            action_cnt=5,
            restore_vars=model_path)

        self.sample_action = self.learner.sample_action

    def overly(self, cur_state, avg_state):
        new_state = [0, 0, 0, 0]
        new_state[0] = cur_state[0] * self.phi + avg_state[0] * (1.0 - self.phi)
        new_state[1] = cur_state[1] * self.phi + avg_state[1] * (1.0 - self.phi)
        new_state[2] = cur_state[2] * self.phi + avg_state[2] * (1.0 - self.phi)
        new_state[3] = cur_state[3] * self.phi + avg_state[3] * (1.0 - self.phi)
        return new_state

    # def GetAvgState(self):
    #     return [self.delay / self.client_num,
    #             self.delivery_rate / self.client_num,
    #             self.send_rate / self.client_num,
    #             self.cwnd / self.client_num]

    def GetExplorationAction(self, state, context):
        with lock:
            cur_state = [state.delay, state.delivery_rate, state.send_rate, state.cwnd]
            avg_state = self.SyncFile(self.pre_state, cur_state, False)
            port = state.port

            input_state = self.overly(cur_state, avg_state)

            action = self.sample_action(input_state)
            op, val = self.action_mapping[action]

            target_cwnd = apply_op(op, input_state[3], val)

            # self.cwnd += target_cwnd - self.client_states[port][3]
            # self.client_states[port][3] = target_cwnd
            self.pre_state = cur_state
            print "target_cwnd: " + str(target_cwnd)
            return indigo_pb2.Action(action=target_cwnd)

    # def update_states(self, state):
    #     port = state.port
    #     if port not in self.client_states:
    #         self.client_num += 1
    #
    #     else:
    #         pre_state = self.client_states[port]
    #         self.delay -= pre_state[0]
    #         self.delivery_rate -= pre_state[1]
    #         self.send_rate -= pre_state[2]
    #         self.cwnd -= pre_state[3]
    #
    #     self.client_states[port] = [state.delay, state.delivery_rate, state.send_rate, state.cwnd]
    #     self.delay += state.delay
    #     self.delivery_rate += state.delivery_rate
    #     self.send_rate += state.send_rate
    #     self.cwnd += state.cwnd

    def UpdateMetric(self, state, context):

        with lock:
            cur_state = [state.delay, state.delivery_rate, state.send_rate, state.cwnd]
            avg_state = self.SyncFile(self.pre_state, cur_state, False)
            input_state = self.overly(cur_state, avg_state)
            self.pre_state = cur_state
            print input_state
            print cur_state
            return indigo_pb2.State(delay=input_state[0], delivery_rate=input_state[1], send_rate=input_state[2],
                                    cwnd=input_state[3], port=state.port)

    def SyncFile(self, pre_state, state, read=False):
        self.check_file()
        with open('states.txt', 'r') as file:
            # 从文件中读取内容，并按空格分割字符串，然后将其转换为浮点数列表
            avg_states = [float(num) for num in file.readline().split()]
            sender_num = int(file.readline())
        if read:
            return avg_states
        else:
            new_avg_state = [0 for _ in range(4)]
            new_avg_state[0] = avg_states[0] * sender_num + state[0] - pre_state[0]
            new_avg_state[1] = avg_states[1] * sender_num + state[1] - pre_state[1]
            new_avg_state[2] = avg_states[2] * sender_num + state[2] - pre_state[2]
            new_avg_state[3] = avg_states[3] * sender_num + state[3] - pre_state[3]
            if pre_state[0] == 0:
                sender_num += 1
            for i in range(len(new_avg_state)):
                new_avg_state[i] /= sender_num
            with open('states.txt', 'w') as file:
                for num in new_avg_state:
                    # 写入每个浮点数，转换为字符串并用空格分隔
                    file.write("{} ".format(num))
                file.write("\n{}".format(sender_num))
            return new_avg_state

    def check_file(self, file_path="states.txt"):
        if os.path.exists(file_path):
            return
        data = """0 0 0 0\n0"""
        with open(file_path, 'w') as file:
            file.write(data)


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
