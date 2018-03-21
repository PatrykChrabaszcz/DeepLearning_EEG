from hpbandster.config_generators.bohb import BOHB
import logging
import os
import pickle
import json


# Initialize logging
logger = logging.getLogger(__name__)


class ConfigGenerator(BOHB):
    """
    Wrapper around BOHB from HpBanSter [https://github.com/automl/HpBandSter]
    We add arguments to the CLI options and log some information each time new result is received.
    This class keeps track of already evaluated configurations for different budgets. When sufficient number of points
    is recorded it will build a Bayesian Optimization model and sample new configurations using that model.
    """
    @staticmethod
    def add_arguments(parser):
        parser.section('config_generator')
        parser.add_argument("min_points_in_model", type=int, default=64,
                            help="Minimal number of points for a given budget that are required to start building a "
                                 "model used for Bayesian Optimization")
        parser.add_argument("top_n_percent", type=int, default=15,
                            help="Percentage of top solutions used as good.")
        parser.add_argument("num_samples", type=int, default=81,
                            help="How many samples are used to optimize EI (Expected Improvement).")
        parser.add_argument("random_fraction", type=float, default=1./3,
                            help="To guarantee exploration, we will draw random samples from the configuration space.")
        parser.add_argument("bandwidth_factor", type=int, default=3,
                            help="HpBandSter samples from wider KDE (Kernel Density Estimator) to keep diversity.")
        parser.add_argument("min_bandwidth", type=float, default=0.001,
                            help="When all good samples have the same value KDE will have bandwidth of 0. "
                                 "Force minimum bandwidth to diversify samples.")

        return parser

    def __init__(self, config_space, working_dir, min_points_in_model, top_n_percent, num_samples, random_fraction,
                 bandwidth_factor, min_bandwidth, **kwargs):
        """
        Args:
            config_space: ConfigSpace object. Contains declaration of all parameters that should be optimized.
                Usually derived from the .pcs files.
            working_dir: Directory for logs from the current experiment.
        """
        super().__init__(configspace=config_space, min_points_in_model=min_points_in_model, top_n_percent=top_n_percent,
                         num_samples=num_samples, random_fraction=random_fraction, bandwidth_factor=bandwidth_factor,
                         min_bandwidth=min_bandwidth, logger=logger)

        self.working_dir = working_dir

        # Try to restore the run
        try:
            logger.info('Check if there are any results from the previous run.')
            with open(os.path.join(self.working_dir, 'configs.p'), 'rb') as f:
                self.configs = pickle.load(f)

            with open(os.path.join(self.working_dir, 'losses.p'), 'rb') as f:
                self.losses = pickle.load(f)

            assert len(self.configs) == len(self.losses), 'Length of configs and losses do not match'
            logger.info('Found %d already evaluated configurations' % len(self.configs))

        except FileNotFoundError:
            logger.warning('Did not find any information about previous BO runs in the current working dir %s.'
                           'Starts from scratch.' % self.working_dir)

    # Maybe do something more?
    def get_config(self, budget):
        config, info_dict = super().get_config(budget)

        return config, info_dict

    # Saves results after each new job. Maybe it can be recovered from the things HpBandSter saves internally (?)
    def new_result(self, job):
        logger.debug('New results received')
        super().new_result(job)

        with open(os.path.join(self.working_dir, 'configs.p'), 'wb') as f:
            logger.debug('Saving configs')
            pickle.dump(self.configs, f)

        with open(os.path.join(self.working_dir, 'losses.p'), 'wb') as f:
            pickle.dump(self.losses, f)
            logger.debug('Saving losses')

        if job.result is not None:
            loss = job.result["loss"]
            info = job.result["info"]
            budget = job.kwargs["budget"]
            config = job.kwargs["config"]

            logger.info('Received new result, loss %s, budget %s' % (loss, budget))

            logger.info(json.dumps(info, indent=2, sort_keys=True))
            logger.info('Config')
            logger.info(json.dumps(config, indent=2, sort_keys=True))
        else:
            logger.warning('Received None result for one of the jobs')
