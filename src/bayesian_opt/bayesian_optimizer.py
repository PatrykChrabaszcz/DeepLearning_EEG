from hpbandster import HB_master
from src.bayesian_opt.config_generator import ConfigGenerator
from hpbandster.HB_iteration import SuccessiveHalving
from hpbandster.distributed.utils import start_local_nameserver
from hpbandster.HB_master import HpBandSter
from time import sleep
import logging
import json
import os


# Initialize logging
logger = logging.getLogger(__name__)


class BayesianOptimizer(object):

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--working_dir", type=str, default='',
                            help="Directory for results and other important stuff.")
        parser.add_argument("--n_iterations", type=int, default=100,
                            help="HPBandSter parameter.")
        parser.add_argument("--run_id", type=str, default='0',
                            help="HPBandSter parameter.")
        parser.add_argument("--eta", type=int, default=3,
                            help="HPBandSter parameter.")
        parser.add_argument("--min_budget", type=float, default=1,
                            help="HPBandSter parameter.")
        parser.add_argument("--max_budget", type=int, default=81,
                            help="HPBandSter parameter.")
        parser.add_argument("--ping_interval", type=int, default=60,
                            help="HPBandSter parameter.")
        parser.add_argument("--nic_name", type=str, default='eth0',
                            help="Network interface card used for Pyro4.")
        return parser

    def __init__(self, config_space,  working_dir, n_iterations, run_id, eta, min_budget, max_budget, ping_interval,
                 nic_name, **kwargs):
        self.config_space = config_space
        self.config_generator = ConfigGenerator(config_space, working_dir=working_dir, **kwargs)

        self.working_dir = working_dir
        self.pyro_conf_file = os.path.join(self.working_dir, 'pyro.conf')

        self.n_iterations = n_iterations
        self.min_budget = min_budget
        self.max_budget = max_budget
        self.eta = eta
        self.ping_interval = ping_interval
        self.run_id = run_id
        self.nic_name = nic_name

        self.optimizer = None
        self.optimizer_thread = None
        self.iteration_class = SuccessiveHalving

    def __enter__(self):
        ns_host, ns_port = start_local_nameserver(nic_name=self.nic_name)
        logger.info('Starting nameserver with %s %s' % (ns_host, ns_port))

        if os.path.exists(self.pyro_conf_file):
            raise RuntimeError('Pyro conf file already exists: %s . '
                               'There is a possibility that some workers already started '
                               'with a wrong conf file!' %(self.pyro_conf_file))

        with open(self.pyro_conf_file, 'w') as f:
            json.dump({'ns_host': ns_host, 'ns_port': ns_port}, f, sort_keys=True, indent=2)

        hpbandster_logger = logging.getLogger('HPBandSter')
        hpbandster_logger.setLevel(logging.WARNING)
        self.optimizer = HpBandSter(run_id=self.run_id, config_generator=self.config_generator, job_queue_sizes=(0, 1),
                                    dynamic_queue_size=True, nameserver=ns_host, ns_port=ns_port, host=ns_host,
                                    min_budget=self.min_budget, max_budget=self.max_budget, eta=self.eta,
                                    ping_interval=self.ping_interval, working_directory=self.working_dir,
                                    logger=hpbandster_logger)
        # Because HpBandSter does not forward logger we need to do it here, Not nice:
        self.optimizer.dispatcher.logger.setLevel(logging.WARNING)
        
        return self

    def run(self):
        return self.optimizer.run(n_iterations=self.n_iterations, min_n_workers=1,
                                  iteration_class=self.iteration_class)

    def __exit__(self, exc_type, exc_value, traceback):
        os.remove(self.pyro_conf_file)

    def shutdown(self):
        self.optimizer.shutdown()

    @staticmethod
    def find_name_server(working_dir, num_tries=60, interval=1):
        conf_file = os.path.join(working_dir, 'pyro.conf')

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
