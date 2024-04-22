import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers, rnn


class ActorCriticNetwork(object):
    def __init__(self, state_dim, action_cnt):
        self.states = tf.placeholder(tf.float32, [None, state_dim])

        actor_h1 = layers.relu(self.states, 8)
        actor_h2 = layers.relu(actor_h1, 8)
        self.action_scores = layers.linear(actor_h2, action_cnt)
        self.action_probs = tf.nn.softmax(self.action_scores)

        critic_h1 = layers.relu(self.states, 8)
        critic_h2 = layers.relu(critic_h1, 8)
        self.state_values = tf.reshape(layers.linear(critic_h2, 1), [-1])

        self.trainable_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)


class ActorCriticLSTM(object):
    def __init__(self, state_dim, action_cnt):
        self.states = tf.placeholder(tf.float32, [None, state_dim])
        self.indices = tf.placeholder(tf.int32, [None])
        rnn_in = tf.expand_dims(self.states, [0])  # shape=(1, ?, state_dim)

        lstm_layers = 2
        lstm_state_dim = 256
        lstm_cell_list = []
        for i in xrange(lstm_layers):
            lstm_cell_list.append(rnn.BasicLSTMCell(lstm_state_dim))
        stacked_cell = rnn.MultiRNNCell(lstm_cell_list)

        rnn_out, lstm_state_out = tf.nn.dynamic_rnn(
            stacked_cell, rnn_in, dtype=tf.float32)  #, initial_state=lstm_state_in)

        self.lstm_state_out = []
        for i in xrange(lstm_layers):
            self.lstm_state_out.append(
                (lstm_state_out[i].c, lstm_state_out[i].h))
        # state output: ((c1, h1), (c2, h2))
        self.lstm_state_out = tuple(self.lstm_state_out)

        # output: shape=(?, lstm_state_dim)
        output = tf.reshape(rnn_out, [-1, lstm_state_dim])
        output = tf.gather(output, self.indices)

        # actor
        actor_h1 = layers.relu(output, 64)
        self.action_scores = layers.linear(actor_h1, action_cnt)
        self.action_probs = tf.nn.softmax(self.action_scores)

        # critic
        critic_h1 = layers.relu(output, 64)
        self.state_values = tf.reshape(layers.linear(critic_h1, 1), [-1])

        self.trainable_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
