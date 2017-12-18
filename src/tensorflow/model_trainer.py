import tensorflow as tf
from src.core.metrics import Metrics
import time
import os
from src.tensorflow.utils import rnn_placeholders


class ModelTrainer:
    def __init__(self, model, train_dr, test_dr, sequence_size):
        self.model = model
        self.train_dr = train_dr
        self.test_dr = test_dr
        self.metrics = Metrics()

        self.global_step = tf.Variable(0, trainable=False)

        self.sequence_size = sequence_size

        self.input_placeholder = tf.placeholder(tf.float32, shape=[None, self.sequence_size, self.train_dr.input_dim])
        self.target_placeholder = tf.placeholder(tf.float32, shape=[None, 1])

        self.state_placeholder = model.state_placeholder()

        state_placeholder = tf.placeholder(tf.float32, [num_layers, 2, batch_size, state_size])
        l = tf.unpack(state_placeholder, axis=0)
        rnn_tuple_state = tuple(
            [tf.nn.rnn_cell.LSTMStateTuple(l[idx][0], l[idx][1])
             for idx in range(num_layers)]
        )


        print('Yoho')
        self.prediction_op = self.model.forward(self.input_placeholder, self.state_placeholder)

        self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.prediction_op[0],
                                                            labels=self.target_placeholder)

        print('Yoho')
        self.optimization_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss, self.global_step)

        self.sv = tf.train.Supervisor(logdir=os.path.join('logs', time.strftime("%Y%m%d-%H%M%S")), summary_op=None,
                                      global_step=self.global_step, save_model_secs=1200)

        print('Evaluate hidden')
        with self.sv.managed_session() as sess:
            hidden = sess.run(self.state_placeholder)

        print('Hidden')
        print(hidden)

        self.train_dr.initialize_epoch(randomize=True, sequence_size=self.sequence_size,
                                       initial_state=hidden)
        self.test_dr.initialize_epoch(randomize=False, sequence_size=self.sequence_size,
                                      initial_state=hidden)

        self.train_dr.start_readers()
        self.test_dr.start_readers()

    def process_one_epoch(self, train=True):
        dr = self.train_dr if train else self.test_dr
        total_loss = 0
        iteration = 0

        try:
            with self.sv.managed_session() as sess:
                while True:
                    batch, labels, ids = dr.get_batch()
                    hidden = self.model.import_hidden(dr.get_states(ids), cuda=True)

                    ops = [self.prediction_op, self.loss]

                    if train:
                        ops.append(self.optimization_op)

                    r = sess.run(ops, feed_dict={self.input_placeholder: batch,
                                                 self.target_placeholder: labels,
                                                 self.state_placeholder: hidden})

                    print(len(r))
                    print(r)
                    prediction, hidden, loss = r[:3]

                    self.metrics.append_results(ids, prediction, labels, train=train)

                    dr.set_states(ids, self.model.export_hidden(hidden))

                    iteration += 1
                    if iteration % 100 is 0:
                        print('Iterations done %d' % iteration)

        except IndexError:
            print('%d Iterations in this epoch' % iteration)

            if iteration > 0:
                print('Loss %g' % (total_loss / iteration))

            self.metrics.finish_epoch(train=train)

            dr.initialize_epoch(randomize=train, sequence_size=self.sequence_size,
                                initial_state=self.model.initial_hidden())

