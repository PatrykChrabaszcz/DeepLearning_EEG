from src.data_reading.data_reader import SequenceDataReader
from src.dl_core.metrics import create_metrics
from src.utils import Stats
import logging


logger = logging.getLogger(__name__)


class ModelTrainerBase:
    _objective_types = ['MeanSquaredError', 'CrossEntropy', 'L1Loss']

    objective_types = [o + '_all' for o in _objective_types].extend([o + '_last' for o in _objective_types])

    def __init__(self, model, lr, l2_decay, objective_type, optimizer, **kwargs):
        self.model = model
        self.learning_rate = lr
        self.weight_decay = l2_decay
        self.objective_type = objective_type
        self.optimizer = optimizer

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--lr", type=float,  dest='lr', default=0.001,
                            help="Learning rate used for training.")
        parser.add_argument("--l2_decay", type=float, dest='l2_decay', default=0.0,
                            help="L2 regularization coefficient.")
        parser.add_argument("--objective_type", type=str, dest='objective_type', choices=ModelTrainerBase.objective_types,
                            default='CrossEntropy_last',
                            help="Whether loss is propagated from all timestamps or just from the last one.")
        parser.add_argument("--iterations_per_epoch", type=int, dest='iterations_per_epoch', default=0,
                            help="If greater than 0 then this many iterations will correspond to one epoch.")
        parser.add_argument("--optimizer", type=str, choices=['Adam', 'SGD'], dest='optimizer', default='Adam',
                            help="Optimizer that is used to update the weights.")

    # If iterations is None then run until EpochDone exception is emitted
    def process_one_epoch(self, forget_state, sequence_size, data_reader, randomize=False, update=False, iterations=None):
        dr = data_reader
        dr.initialize_epoch(sequence_size=sequence_size, randomize=randomize)

        time_stats = Stats('Time Statistics')
        get_batch_stats = time_stats.create_child_stats('Get Batch')
        one_iteration_stats = time_stats.create_child_stats('Forward Pass')
        process_metrics_stats = time_stats.create_child_stats('Process Metrics')
        save_states_stats = time_stats.create_child_stats('Save States')

        metrics = create_metrics(objective_type=self.objective_type)
        iteration = 0
        try:
            with time_stats:
                while True:
                    if iterations is not None and iteration == iterations:
                        raise SequenceDataReader.EpochDone

                    with get_batch_stats:
                        ids, batch, time, labels, contexts = dr.get_batch()
                        labels = labels[:, self.model.offset_size(sequence_size):]
                        hidden = self.model.import_state(dr.get_states(ids, forget=forget_state))

                    with one_iteration_stats:
                        outputs, hidden, loss = self._one_iteration(batch=batch, time=time, hidden=hidden,
                                                                    labels=labels, context=contexts, update=update)
                    with process_metrics_stats:
                        self._gather_results(ids, outputs, labels, loss, batch_size=len(batch), metrics=metrics)

                    with save_states_stats:
                        dr.set_states(ids, self.model.export_state(hidden))

                    iteration += 1
                    if iteration % 100 is 0:
                        logger.debug('Iterations done %d' % iteration)

        except SequenceDataReader.EpochDone:
            logger.info('%d Iterations in this epoch' % iteration)
            return metrics

    def _one_iteration(self, batch, time, hidden, labels, context, update=False):
        raise NotImplementedError

    def _gather_results(self, ids, outputs, labels, loss, batch_size, metrics):
        raise NotImplementedError