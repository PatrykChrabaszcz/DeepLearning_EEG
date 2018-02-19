import logging
from src.experiment import Experiment

# Initialize logging
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    logger.info('Create experiment')
    experiment = Experiment()

    logger.info('Start hyper-parameter search')
    experiment.run_search_experiment()
