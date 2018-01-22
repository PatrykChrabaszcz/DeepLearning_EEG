import logging
from time import strftime

import click

from src.data_reading.data_reader import SequenceDataReader
from src.result_logger import ResultsLogger
from src.utils import Stats, setup_logging
from tutorial.data_reader import TutorialDataReader
from tutorial.model import TutorialRNN
from tutorial.model_trainer import ModelTrainer

# Initialize logging
logger = logging.getLogger(__name__)
setup_logging()


@click.command()
@click.option('--sequence_size', default=15)
@click.option('--batch_size', default=64)
@click.option('--learning_rate', default=0.001)
@click.option('--train_readers', default=2)
@click.option('--test_readers', default=2)
@click.option('--forget_state', is_flag=True)
def main(sequence_size, batch_size, learning_rate, train_readers, test_readers, forget_state):

    logger.info('Will use PyTorch backend')

    time_string = strftime("%Y%m%d-%H%M%S")
    experiment_name = 'SeqSize(%s)_LR(%s)_ForgetState(%s)_%s' % \
                      (sequence_size, learning_rate, forget_state, time_string)

    logger.info('Experiment name: %s' % experiment_name)

    model = TutorialRNN(1, 32, 3, 1, cell_type='LSTM')

    # Initialize data readers
    train_dr = TutorialDataReader(readers_count=train_readers, batch_size=batch_size,
                                  state_initializer=model.initial_state)
    test_dr = TutorialDataReader(readers_count=test_readers, batch_size=batch_size,
                                 state_initializer=model.initial_state, data_type=SequenceDataReader.Validation_Data)

    model_trainer = ModelTrainer(model, learning_rate, 0.0001, train_dr, test_dr, sequence_size,
                                 loss_type='classification_last')
    results_logger = ResultsLogger(experiment_name=experiment_name)

    logger.info('Number of parameters in the model %d' % model.count_params())
    epoch = 1
    try:
        import time
        while True:
            with Stats('Training Epoch Took'):
                results = model_trainer.process_one_epoch(train=True, forget_state=forget_state)
                results_logger.log_metrics(results, epoch, train=True, forget_state=forget_state)
                print('Training results')
                print(results)

            with Stats('Validation Epoch Forget Took'):
                results = model_trainer.process_one_epoch(train=False, forget_state=True)
                print('Validation results Forget State')
                print(results)
                results_logger.log_metrics(results, epoch, train=False, forget_state=True)
            with Stats('Validation Epoch No Forget Took'):
                results = model_trainer.process_one_epoch(train=False, forget_state=False)
                print('Validation results Remember State')
                print(results)
                results_logger.log_metrics(results, epoch, train=False, forget_state=False)
            epoch += 1

            model.save_model('models/tmp.model')

    except (InterruptedError, KeyboardInterrupt):
        train_dr.stop_readers()
        test_dr.stop_readers()


if __name__ == '__main__':
    main()
