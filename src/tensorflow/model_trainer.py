import tensorflow as tf
from src.core.metrics import Metrics
import time
import os
import numpy as np
from src.utils import Stats


class ModelTrainer:
    def __init__(self, model, learning_rate, train_dr, test_dr, sequence_size, forget_state):
        self.model = model
        self.train_dr = train_dr
        self.test_dr = test_dr
        self.forget_state = forget_state
        self.metrics = Metrics()

        self.global_step = tf.Variable(0, trainable=False)

        self.sequence_size = sequence_size

        self.input_placeholder = tf.placeholder(tf.float32, shape=[None, self.sequence_size, self.train_dr.input_dim])
        self.target_placeholder = tf.placeholder(tf.int32, shape=[None])

        self.state_placeholder, self.state = self.model.state_placeholders()
        self.forward_op = self.model.forward(self.input_placeholder, self.state)

        self.prediction = self.forward_op[0]

        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.prediction,
                                                                                  labels=self.target_placeholder))

        self.optimization_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss, self.global_step)

        self.sv = tf.train.Supervisor(logdir=os.path.join('logs', time.strftime("%Y%m%d-%H%M%S")), summary_op=None,
                                      global_step=self.global_step, save_model_secs=1200)

        self.train_dr.initialize_epoch(randomize=True, sequence_size=self.sequence_size)
        self.test_dr.initialize_epoch(randomize=False, sequence_size=self.sequence_size)

        self.train_dr.start_readers()
        self.test_dr.start_readers()

    def process_one_epoch(self, train=True):
        dr = self.train_dr if train else self.test_dr
        total_loss = 0
        iteration = 0
        with self.sv.managed_session() as sess:
            try:
                while True:
                    batch, time, labels, ids = dr.get_batch()
                    hidden = self.model.import_state(dr.get_states(ids, forget=self.forget_state), cuda=True)

                    ops = [self.forward_op, self.loss]

                    if train:
                        ops.append(self.optimization_op)

                    feed_dict = self.get_placeholder_dict(hidden)
                    feed_dict[self.input_placeholder] = batch
                    feed_dict[self.target_placeholder] = labels

                    r = sess.run(ops, feed_dict)

                    (prediction, hidden), loss = r[:2]
                    total_loss += loss

                    self.metrics.append_results(ids, prediction, labels, train=train)

                    dr.set_states(ids, self.model.export_state(hidden))

                    iteration += 1
                    if iteration % 100 is 0:
                        print('Iterations done %d' % iteration)

            except IndexError:
                print('%d Iterations in this epoch' % iteration)

                result = self.metrics.finish_epoch(train=train)

                dr.initialize_epoch(randomize=train, sequence_size=self.sequence_size)

                return result

    def get_placeholder_dict(self, state):
        d = dict()
        for p, s in zip(self.state_placeholder, state):
            d[p] = s

        return d
