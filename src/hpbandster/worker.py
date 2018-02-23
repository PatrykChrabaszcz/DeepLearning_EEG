from hpbandster.distributed.worker import Worker as HpBandSterWorker
from src.data_reading.data_reader import SequenceDataReader
from hpbandster.distributed.utils import nic_name_to_host
from src.dl_core.metrics import average_metrics_results
from time import sleep
import Pyro4
import logging
import json
import os


# Initialize logging
logger = logging.getLogger(__name__)
hpbandster_logger = logging.getLogger('HPBandSter')
hpbandster_logger.setLevel(logging.WARNING)


class Worker(HpBandSterWorker):
    def __init__(self, train_manager, budget_decoder, experiment_args, working_dir, nic_name, run_id, **kwargs):
        logger.info('Creating worker for distributed computation.')

        self.train_manager = train_manager
        self.budget_decoder = budget_decoder
        self.experiment_args = experiment_args
        self.working_dir = working_dir

        ns_name, ns_port = self._search_for_name_server()
        logger.info('Worker found nameserver %s, %s' % (ns_name, ns_port))

        host = nic_name_to_host(nic_name)
        logger.info('Worker will try to run on a host %s' % host)

        super().__init__(run_id=run_id, nameserver=ns_name, ns_port=ns_port, host=host,
                         logger=hpbandster_logger)

    def compute(self, config, budget, **kwargs):
        logger.info('Worker: Starting computation for budget %s ' % budget)

        if config is not None:
            raise RuntimeError('Worker received config that is None in compute(...)')

        adjusted_experiment_args = self.budget_decoder.adjusted_arguments(self.experiment_args, budget)

        # Each evaluation can mean multiple folds of CV
        result_list = []
        for experiment_args in adjusted_experiment_args:
            self.train_manager.init_new_log_dir()
            self.train_manager.train(experiment_args)
            valid_metrics = self.train_manager.validate(experiment_args, log_dir=self.train_manager.log_dir,
                                                        data_type=SequenceDataReader.Validation_Data)
            result_list.append(valid_metrics.get_summarized_results())

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

    def _search_for_name_server(self, num_tries=60, interval=1):
        """
        Will try to find pyro.conf file in the current working_dir and extract ns_name and ns_port parameters.
        Will update internal parameters if values for the current experiment were found
        :param num_tries:
        :param interval:
        :return:
        """

        conf_file = os.path.join(self.working_dir, 'pyro.conf')

        user_notified = False
        for i in range(num_tries):
            try:
                with open(conf_file, 'r') as f:
                    d = json.load(f)
                logger.debug('Found nameserver info %s' % d)
                return d['ns_host'], d['ns_port']

            except FileNotFoundError:
                if not user_notified:
                    logger.info('Config file not found. Waiting for the master node to start')
                    user_notified = True
                sleep(interval)

        raise RuntimeError("Could not find the nameserver information after %d tries, aborting!" % num_tries)
