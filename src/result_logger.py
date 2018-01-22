import tensorflow as tf
from os import makedirs
import json


class ResultsLogger:
    def __init__(self, reader_args, model_args, trainer_args, experiment_name):
        experiment_path = 'logs/%s' % experiment_name
        makedirs(experiment_path)
        with open('%s/reader_args.json' % experiment_path, 'w') as f:
            json.dump(reader_args, f, indent=4)
        with open('%s/model_args.json' % experiment_path, 'w') as f:
            json.dump(model_args, f, indent=4)
        with open('%s/trainer_args.json' % experiment_path, 'w') as f:
            json.dump(trainer_args, f, indent=4)

        self.summary_writers = {
            'train': {
                'forget': tf.summary.FileWriter(logdir='%s/Train_forget_state' % experiment_path),
                'remember': tf.summary.FileWriter(logdir='%s/Train_remember_state' % experiment_path)
            },
            'test': {
                'forget': tf.summary.FileWriter(logdir='%s/Test_forget_state' % experiment_path),
                'remember': tf.summary.FileWriter(logdir='%s/Test_remember_state' % experiment_path)
            }
        }

    def log_metrics(self, metrics_dict, step, forget_state, train=True):
        summary_writer = self.summary_writers['train'] if train else self.summary_writers['test']
        summary_writer = summary_writer['forget'] if forget_state else summary_writer['remember']

        for name, value in metrics_dict.items():
            summary = tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=value)])
            summary_writer.add_summary(summary, step)

        summary_writer.flush()
