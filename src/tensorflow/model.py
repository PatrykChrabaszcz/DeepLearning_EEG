from tensorflow.contrib.rnn import BasicLSTMCell, GRUCell, LSTMBlockFusedCell, LSTMBlockCell, PhasedLSTMCell
from tensorflow.contrib.cudnn_rnn import CudnnLSTM
from tensorflow.python.util.nest import flatten
from tensorflow.contrib.layers import fully_connected
import tensorflow as tf
import numpy as np


class SimpleRNN:
    def __init__(self, input_size, hidden_size, num_layers=3, num_classes=2):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.cell = BasicLSTMCell(num_units=self.hidden_size)
        self.network = tf.contrib.rnn.MultiRNNCell([self.cell] * self.num_layers)

    def forward(self, x, hidden):
        outputs, hidden = tf.nn.dynamic_rnn(self.network, x, initial_state=hidden, dtype=tf.float32)
        last_output = outputs[:, -1, :]
        output = fully_connected(last_output, num_outputs=2, activation_fn=None)
        return output, hidden

    def initial_state(self):
        state = []
        for _ in range(self.num_layers):
            c = np.random.normal(0, 1.0, [1, self.hidden_size])
            h = np.random.normal(0, 1.0, [1, self.hidden_size])
            state.extend([c, h])

        return state

    def state_placeholders(self):
        state = []
        phs = []
        for _ in range(self.num_layers):
            c = tf.placeholder(dtype=tf.float32, shape=[None, self.hidden_size])
            h = tf.placeholder(dtype=tf.float32, shape=[None, self.hidden_size])
            phs.append(c)
            phs.append(h)
            state.append(tf.contrib.rnn.LSTMStateTuple(c, h))

        return phs, tuple(state)

    @staticmethod
    def export_state(states):
        flatten_states = flatten(states)
        batch_size = flatten_states[0].shape[0]

        exported_states = []
        for i in range(batch_size):
            exported_states.append([f_s[np.newaxis, i] for f_s in flatten_states])

        return exported_states

    @staticmethod
    def import_state(states, cuda=True):
        state_tensor_count = len(states[0])
        imported_states = []
        for i in range(state_tensor_count):
            imported_states.append(np.concatenate([s[i] for s in states]))

        return tuple(imported_states)

    @staticmethod
    def count_params():
        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        return total_parameters
