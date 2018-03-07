import logging
from src.experiment import Experiment

# Initialize logging
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    logger.info('Create experiment')
    experiment = Experiment()

    experiment.main()
