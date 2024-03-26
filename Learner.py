import argparse
import project_root
import numpy as np
import tensorflow as tf
from os import path
from models import ActorCriticLSTM
from a3c import ewma
from helpers.helpers import normalize, one_hot, softmax


class Learner(object):
    def __init__(self, state_dim, action_cnt, restore_vars):
        with tf.variable_scope('local'):
            self.pi = ActorCriticLSTM(
                state_dim=state_dim, action_cnt=action_cnt)
            # # save the current LSTM state of local network
            # self.lstm_state = self.pi.lstm_state_init

        self.session = tf.Session()

        # restore saved variables
        saver = tf.train.Saver(self.pi.trainable_vars)
        saver.restore(self.session, restore_vars)

        # init the remaining vars, especially those created by optimizer
        uninit_vars = set(tf.global_variables()) - set(self.pi.trainable_vars)
        self.session.run(tf.variables_initializer(uninit_vars))

    def sample_action(self, step_state_buf):
        # ravel() is a faster flatten()
        flat_step_state_buf = np.asarray(step_state_buf, dtype=np.float32).ravel()

        # state = EWMA of past step
        ewma_delay = ewma(flat_step_state_buf, 3)

        ops_to_run = [self.pi.action_probs]#, self.pi.lstm_state_out]
        feed_dict = {
            self.pi.states: [ewma_delay],
            self.pi.indices: [0],
            # self.pi.lstm_state_in: self.lstm_state,
        }

        ret = self.session.run(ops_to_run, feed_dict)
        action_probs = ret#, lstm_state_out = ret

        action = np.argmax(action_probs)
        # action = np.argmax(np.random.multinomial(1, action_probs[0] - 1e-5))
        # self.lstm_state = lstm_state_out
        return action