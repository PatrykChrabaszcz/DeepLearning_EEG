import tensorflow as tf


class ResultsLogger:
    def __init__(self, experiment_name):
        self.summary_writer_train = tf.summary.FileWriter(logdir='logs/Train_%s' % experiment_name)
        self.summary_writer_test = tf.summary.FileWriter(logdir='logs/Test_%s' % experiment_name)

    def log_metrics(self, metrics_dict, step, train=True):
        summary_writer = self.summary_writer_train if train else self.summary_writer_test
        for name, value in metrics_dict.items():
            summary = tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=value)])
            summary_writer.add_summary(summary, step)

        summary_writer.flush()
