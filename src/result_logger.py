import tensorflow as tf
from os import makedirs
import configparser


class ResultsLogger:
    def __init__(self, log_dir, worker_args, reader_args, model_args, trainer_args, experiment_name):
        self.experiment_path = '%s/%s' % (log_dir, experiment_name)
        makedirs(self.experiment_path)

        # Save .ini file to make it possible to restore experiment
        parser = configparser.ConfigParser()

        for name, d in (('worker', worker_args), ('model', model_args),
                        ('reader', reader_args), ('trainer', trainer_args)):
            parser.add_section(name)
            for key, value in d.items():
                parser.set(name, key, str(value))

        with open('%s/%s' % (self.experiment_path, 'config.ini'), 'w') as f:
            parser.write(f)

        # Tensorflow logging
        self.summary_writers = {
            'train': tf.summary.FileWriter(logdir='%s/Train' % self.experiment_path),
            'validation': tf.summary.FileWriter(logdir='%s/Valid' % self.experiment_path),
            'test': tf.summary.FileWriter(logdir='%s/Test' % self.experiment_path),
        }

    def log_metrics(self, metrics_dict, data_type='train'):
        # Find out correct proper summary writer object
        summary_writer = self.summary_writers[data_type]

        for name, value in metrics_dict.items():
            summary = tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=value)])
            summary_writer.add_summary(summary, 0)

        summary_writer.flush()
