from src.experiment_arguments import ExperimentArguments
from src.hpbandster.budget_decoder import FullBudgetDecoder
from src.experiment import Experiment
import logging
import json


logger = logging.getLogger(__name__)


def main():
    # Experiment will read all arguments from the .ini file and command line interface (CLI).
    experiment = Experiment()

    train_manager = experiment.train_manager
    experiment_arguments = experiment.experiment_arguments

    # Sample random configuration
    config_space = ExperimentArguments.read_configuration_space(experiment_arguments.config_space_file)
    random_config = config_space.sample_configuration()

    # Update experiment arguments using random configuration
    experiment_arguments = experiment_arguments.updated_with_configuration(random_config)

    # Run for different budgets
    full_budget_decoder = FullBudgetDecoder()
    adjusted_arguments_list = full_budget_decoder.adjusted_arguments(experiment_arguments, budget=None)

    for experiment_args in adjusted_arguments_list:
        # Initialize a new directory for each training run
        experiment_args.run_log_folder = train_manager.get_unique_dir()

        train_metrics = train_manager.train(experiment_args)
        valid_metrics = train_manager.validate(experiment_args, data_type=experiment_arguments.validation_data_type)

        # Print for the user
        logger.info('Train Metrics:')
        logger.info(json.dumps(train_metrics.get_summarized_results(), indent=2, sort_keys=True))
        logger.info('%s Metrics:' % experiment_arguments.validation_data_type.title())
        logger.info(json.dumps(valid_metrics.get_summarized_results(), indent=2, sort_keys=True))
