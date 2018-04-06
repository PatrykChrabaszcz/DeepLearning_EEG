from src.data_reading.data_reader import SequenceDataReader
from src.experiment import Experiment
import logging
import json
import os

logger = logging.getLogger(__name__)


def main():
    # Experiment will read all arguments from the .ini file and command line interface (CLI).
    experiment = Experiment()

    train_manager = experiment.train_manager
    experiment_arguments = experiment.experiment_arguments

    # Get important arguments
    validation_data_type = experiment_arguments.validation_data_type
    run_log_folder = experiment_arguments.run_log_folder

    if run_log_folder == "":
        raise RuntimeError("Did you forget to specify run_log_folder? No idea which model to validate.")

    # We don't force the user to use specific settings during evaluation but people make mistakes so display
    # warnings when unexpected behavior is spotted.
    if validation_data_type in [SequenceDataReader.Train_Data, SequenceDataReader.Test_Data]:
        logger.warning('Evaluation on %s data set (Are you sure?)' % validation_data_type)

    if experiment_arguments.random_mode != 0:
        logger.warning('Running evaluation, but random mode is not 0 (Are you sure?)')
    if experiment_arguments.continuous != 0:
        logger.warning('Running evaluation, but continuous is not 0 (Are you sure?)')
    if experiment_arguments.forget_state == 1:
        logger.warning('Running evaluation, but forget_state is set to 1 (Are you sure?)')
    if experiment_arguments.balanced == 1:
        logger.warning('Running evaluation, but balanced is set to 1 (Are you sure?)')

    # Do not save metrics in the train folder!
    metrics = train_manager.validate(experiment_arguments, save_metrics=False)

    # Use some other place to store results
    name = ('n%s_k%s' % (experiment_arguments.cv_n, experiment_arguments.cv_k))
    metrics.save(directory=os.path.join('/home/chrabasp/EEG_Result_Val/eval_val_top3_anomaly', name))

    logger.info('%s Metrics:' % validation_data_type)
    logger.info(json.dumps(metrics.get_summarized_results(), indent=2, sort_keys=True))


if __name__ == '__main__':
    logger.info('Start single evaluation')
    main()