from hpbandster.distributed.utils import nic_name_to_host
from hpbandster.distributed.worker import Worker
from datetime import datetime
import logging
import json
import os
import Pyro4
from src.dl_core.metrics import average_metrics_results
from src.bayesian_opt.bayesian_optimizer import BayesianOptimizer

# Initialize logging
logger = logging.getLogger(__name__)


class DLWorker(Worker):
    def __init__(self, ModelClass, ReaderClass, TrainerClass, budget_decoder, experiment_args, nic_name, **kwargs):

        self.ModelClass = ModelClass
        self.ReaderClass = ReaderClass
        self.TrainerClass = TrainerClass
        self.experiment_args = experiment_args
        self.working_dir = experiment_args.working_dir
        self.budget_decoder = budget_decoder

        ns_name, ns_port = BayesianOptimizer.find_name_server(working_dir=self.working_dir)
        logger.info('Worker found nameserver %s, %s' % (ns_name, ns_port))

        host = nic_name_to_host(nic_name)
        logger.info('Worker will try start on a host %s' % host)

        hpbandster_logger = logging.getLogger('HPBandSter')
        hpbandster_logger.setLevel(logging.WARNING)
        super().__init__(experiment_args.run_id, nameserver=ns_name, ns_port=ns_port, host=host,
                         logger=hpbandster_logger)

    @staticmethod
    def save_logs(run_dir, train_metrics, evaluation_metrics, args):
        assert not os.path.exists(run_dir), 'This run directory already exists %s' % run_dir

        os.makedirs(run_dir)
        with open(os.path.join(run_dir, 'arguments.json'), 'w') as f:
            json.dump(args, f, sort_keys=True, indent=2)

        train_metrics.save(os.path.join(run_dir, 'train'))
        evaluation_metrics.save(os.path.join(run_dir, 'eval'))

    def compute(self, config, budget, **kwargs):
        logger.info('Worker: Starting computation for budget %s ' % budget)

        experiment_args = self.experiment_args.updated_with_configuration(config)
        adjusted_args = self.budget_decoder.adjusted_arguments(experiment_args, budget)

        # Each evaluation can mean multiple folds of CV
        result_list = []
        for i, args in enumerate(adjusted_args):
            args = args.get_arguments()
            run_dir = os.path.join(self.working_dir, 'train_logs',
                                   datetime.utcnow().strftime('%Y_%m_%d__%H_%M_%S_%f'))

            context_size = self.ReaderClass.context_size(**args)
            input_size = self.ReaderClass.input_size(**args)
            output_size = self.ReaderClass.output_size(**args)

            model = self.ModelClass(input_size=input_size, output_size=output_size, context_size=context_size,
                                    **args)

            offset_size = model.offset_size(sequence_size=args['initial_sequence_size'])
            logger.info('Data readers will use an offset:  %d' % offset_size)

            train_dr = self.ReaderClass(offset_size=offset_size, allow_smaller_batch=False,
                                        state_initializer=model.initial_state,
                                        data_type=self.ReaderClass.Train_Data,
                                        **args)

            valid_dr = self.ReaderClass(offset_size=offset_size, allow_smaller_batch=True,
                                        state_initializer=model.initial_state,
                                        data_type=self.ReaderClass.Validation_Data,
                                        **args)

            trainer = self.TrainerClass(model=model, **args)

            # Train the model
            train_metrics = trainer.run(data_reader=train_dr, train=True)

            # Evaluate the model
            evaluation_metrics = trainer.run(data_reader=valid_dr, train=False)
            result_list.append(evaluation_metrics.get_summarized_results())

            DLWorker.save_logs(run_dir, train_metrics, evaluation_metrics, args)

        # TODO Aggregate results and return format that is compatible with HPBandSter

        averaged_results = average_metrics_results(result_list)
        logger.info('Computation done, submit results (loss %s)' % averaged_results['loss'])
        return {
            'loss': averaged_results['loss'],
            'info': averaged_results
        }

    @Pyro4.expose
    @Pyro4.oneway
    def shutdown(self):
        logger.info('Shutting down Worker')
        super().shutdown()

