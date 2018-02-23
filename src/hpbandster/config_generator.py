from hpbandster.config_generators.bohb import BOHB
import logging
import numpy as np


# Initialize logging
logger = logging.getLogger(__name__)


class ConfigGenerator(BOHB):
    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--min_points_in_model", type=int, default=64,
                            help="HPBandSter minimum number of points in the model.")
        parser.add_argument("--top_n_percent", type=int, default=15,
                            help="HPBandSter parameter.")
        parser.add_argument("--num_samples", type=int, default=81,
                            help="HPBandSter parameter.")
        parser.add_argument("--random_fraction", type=float, default=1./3,
                            help="HPBandSter parameter.")
        parser.add_argument("--bandwidth_factor", type=int, default=3,
                            help="HPBandSter parameter.")
        return parser

    def __init__(self, config_space, min_points_in_model, top_n_percent, num_samples, random_fraction,
                 bandwidth_factor, working_dir, **kwargs):

        super().__init__(configspace=config_space, directory=working_dir,
                         min_points_in_model=min_points_in_model, top_n_percent=top_n_percent,
                         num_samples=num_samples, random_fraction=random_fraction, bandwidth_factor=bandwidth_factor)

        self.working_dir = working_dir
        # TODO Read self.configs and self.losses

    def new_result(self, job):
        if job.result is not None:
            loss = job.result["loss"]
            info = job.result["info"]
            budget = job.kwargs["budget"]
            config = job.kwargs["config"]

            logger.info('Received new result, loss %s, budget %s' % (loss, budget))
            logger.info(info)
            logger.info('Config')
            logger.info(config)
        else:
            logger.warning('Received None result for one of the jobs')

        super().new_result(job)

        # TODO Save self.configs and self.losses