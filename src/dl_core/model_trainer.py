from src.data_reading.data_reader import SequenceDataReader
from src.dl_core.metrics import create_metrics
from time import time as get_time
from src.utils import Stats
import logging
logger = logging.getLogger(__name__)


class ModelTrainerBase:
    class StopRun:
        """
        Class used to terminate training or evaluation.
        """
        def __init__(self, budget_type, budget, train):
            self.budget_type = budget_type
            self.budget = budget
            self.train = train
            self.start_time = get_time()
            self.iteration = 0
            self.epoch = 0

        def check_after_iteration_done(self):
            self.iteration += 1
            return (self._check_iterations() or self._check_time()) and self.train

        def check_after_epoch_done(self):
            self.epoch += 1
            return self._check_epochs() or self._check_time() or not self.train

        def _check_time(self):
            if self.budget_type != 'minute':
                return False
            minutes_passed = (get_time() - self.start_time) / 60
            return minutes_passed > self.budget

        def _check_iterations(self):
            if self.budget_type != 'iteration':
                return False
            return self.iteration >= self.budget

        def _check_epochs(self):
            if self.budget_type != 'epoch':
                return False
            return self.epoch >= self.budget

    _objective_types = ['MeanSquaredError', 'CrossEntropy', 'L1Loss']

    objective_types = [o + '_all' for o in _objective_types].extend([o + '_last' for o in _objective_types])

    budget_types = ['epoch', 'iteration', 'minute']

    def __init__(self, model, lr, l2_decay, objective_type, budget, budget_type,
                 optimizer, cosine_restarts_decay, **kwargs):
        self.model = model
        self.learning_rate = lr
        self.weight_decay = l2_decay
        self.objective_type = objective_type
        self.budget = budget
        self.budget_type = budget_type
        self.optimizer = optimizer
        self.cosine_restarts_decay = cosine_restarts_decay

    @staticmethod
    def add_arguments(parser):
        parser.section('model_trainer')
        parser.add_argument("lr", type=float, default=0.001,
                            help="Learning rate used for training.")
        parser.add_argument("l2_decay", type=float, default=0.0,
                            help="L2 regularization coefficient.")
        parser.add_argument("objective_type", type=str,
                            choices=ModelTrainerBase.objective_types,
                            default='CrossEntropy_last',
                            help="Whether loss is propagated from all timestamps or just from the last one.")

        parser.add_argument("budget", type=int, default=1,
                            help="Training budget")
        parser.add_argument("budget_type", type=str, default='epoch',
                            choices=ModelTrainerBase.budget_types,
                            help="Type of the training budget.")

        parser.add_argument("optimizer", type=str, choices=['Adam', 'AdamW', 'SGD'],
                            default='AdamW',
                            help="Optimizer that is used to update the weights.")
        parser.add_argument("cosine_restarts_decay", type=int, choices=[0, 1],
                            default=0,
                            help="If set to 1 then will use cosine decay learning rate schedule with restarts.")
        return parser

    def run(self, data_reader, train=True):
        time_stats = Stats('Time Statistics')
        get_batch_stats = time_stats.create_child_stats('Get Batch')
        one_iteration_stats = time_stats.create_child_stats('Forward Pass')
        process_metrics_stats = time_stats.create_child_stats('Process Metrics')
        save_states_stats = time_stats.create_child_stats('Save States')

        metrics = create_metrics(name=data_reader.data_type,
                                 objective_type=self.objective_type,
                                 output_size=data_reader.output_size())

        stop_run = ModelTrainerBase.StopRun(self.budget_type, self.budget, train=train)

        with time_stats:
            # If something happens we need to stop the reader
            try:
                data_reader.start_readers()
                # Throws an exception when epoch is done
                # Multiple epochs
                while True:
                    try:
                        data_reader.initialize_epoch()

                        # Multiple iterations
                        while True:
                            with get_batch_stats:
                                ids, batch, time, labels, contexts = data_reader.get_batch()
                                labels = labels[:, self.model.offset_size(data_reader.sequence_size):]
                                hidden = self.model.import_state(data_reader.get_states(ids))

                            with one_iteration_stats:
                                outputs, hidden, loss = self._one_iteration(batch=batch, time=time, hidden=hidden,
                                                                            labels=labels, context=contexts,
                                                                            update=train)
                            with process_metrics_stats:
                                self._gather_results(ids, outputs, labels, loss, batch_size=len(batch), metrics=metrics)

                            with save_states_stats:
                                data_reader.set_states(ids, self.model.export_state(hidden))

                            if stop_run.check_after_iteration_done():
                                logger.info('Limit reached with %d iterations. Stop training.' % stop_run.iteration)
                                data_reader.stop_readers()
                                return metrics

                            if stop_run.iteration % 100 is 0:
                                name = 'Train' if train else 'Validation'
                                logger.debug('%s iterations done %d, loss %g' % (name, stop_run.iteration,
                                                                                 metrics.get_current_loss()))

                                if metrics.get_current_loss() != metrics.get_current_loss():
                                    raise RuntimeError('Loss is NaN, terminate run')

                    except SequenceDataReader.EpochDone:
                        if stop_run.check_after_epoch_done():
                            logger.info('Limit reached with %s epochs. Stop the run.' % stop_run.epoch)
                            data_reader.stop_readers()
                            return metrics

            except (Exception, KeyboardInterrupt) as e:
                logger.warning('Exception detected, trying to stop training readers ...')
                data_reader.stop_readers()
                # This exception should be caught by a worker and reported to the BO optimizer
                raise e

    def _one_iteration(self, batch, time, hidden, labels, context, update=False):
        raise NotImplementedError

    def _gather_results(self, ids, outputs, labels, loss, batch_size, metrics):
        raise NotImplementedError
