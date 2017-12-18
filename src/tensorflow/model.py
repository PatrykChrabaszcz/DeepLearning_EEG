from tensorflow.contrib.rnn import BasicLSTMCell, static_rnn
from tensorflow.contrib.layers import fully_connected
import tensorflow as tf
import numpy as np

class SimpleRNN:
    def __init__(self, input_size, hidden_size, num_layers=3, num_classes=2):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.cell = BasicLSTMCell(self.hidden_size, forget_bias=1.0, state_is_tuple=False)
        self.network = tf.contrib.rnn.MultiRNNCell([self.cell] * self.num_layers)

    def forward(self, x, hidden):
        outputs, hidden = tf.nn.dynamic_rnn(self.network, x, initial_state=hidden, dtype=tf.float32)
        last_output = outputs[-1]
        output = fully_connected(last_output, num_outputs=2, activation_fn=None)

        return output, hidden

    def state_placeholder(self):
        return tf.placeholder(tf.float32, [None, self.num_layers, 2, self.hidden_size])

    def initial_hidden(self):
        return np.random.normal(0, 1.0, [self.num_layers, 2, self.hidden_size])

    # Converts PyTorch hidden state representation into something that can be saved
    def export_hidden(self, states):
        return states

    # Converts PyTorch hidden state representation into something that can be saved
    def import_hidden(self, states, cuda=True):
        return states

    def count_params(self):
        return 15
