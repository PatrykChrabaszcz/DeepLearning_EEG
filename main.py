from src.data_reader import AnomalyDataReader, SequenceDataReader
from src.utils import Stats, setup_logging
import click
import logging
from src.result_logger import ResultsLogger
from time import strftime

# Initialize logging
logger = logging.getLogger(__name__)
setup_logging()


@click.command()
@click.option('--data_path', type=click.Path(exists=True), required=True)
@click.option('--sequence_size', default=500)
@click.option('--batch_size', default=64)
@click.option('--learning_rate', default=0.001)
@click.option('--train_readers', default=5)
@click.option('--test_readers', default=5)
@click.option('--limit_duration', default=None)
@click.option('--limit_examples', default=None)
@click.option('--backend', default='Pytorch')
@click.option('--forget_state', is_flag=True)
def main(data_path, sequence_size, batch_size, learning_rate, train_readers, test_readers, limit_duration,
         limit_examples, backend, forget_state):
    if limit_examples is not None:
        limit_examples = int(limit_examples)
    if limit_duration is not None:
        limit_duration = int(limit_duration)

    # Initialize backend
    if backend.title() == 'Tensorflow':
        logger.info('Will use Tensorflow backend')
        from src.tensorflow.model import SimpleRNN
        from src.tensorflow.model_trainer import ModelTrainer
    elif backend.title() == 'Pytorch':
        logger.info('Will use PyTorch backend')
        from src.pytorch.model import SimpleRNN
        from src.pytorch.model_trainer import ModelTrainer
    else:
        raise NotImplementedError('Specify backend as Tensorflow or PyTorch')

    time_string = strftime("%Y%m%d-%H%M%S")
    experiment_name = 'Backend(%s)_SeqSize(%s)_LR(%s)_ForgetState(%s)_%s' % \
                      (backend, sequence_size, learning_rate, forget_state, time_string)

    logger.info('Experiment name: %s' % experiment_name)

    model = SimpleRNN(22, 128, 3, 2)

    # Initialize data readers
    train_dr = AnomalyDataReader(data_path, limit_examples, limit_duration,
                                 readers_count=train_readers, batch_size=batch_size,
                                 state_initializer=model.initial_state)
    test_dr = AnomalyDataReader(data_path, limit_examples, limit_duration,
                                readers_count=test_readers, batch_size=batch_size,
                                state_initializer=model.initial_state, data_type=SequenceDataReader.Validation_Data)

    model_trainer = ModelTrainer(model, learning_rate, train_dr, test_dr, sequence_size, forget_state)
    results_logger = ResultsLogger(experiment_name=experiment_name)

    logger.info('Number of parameters in the model %d' % model.count_params())
    epoch = 1
    try:
        import time
        while True:
            logger.info('Start Train Epoch: Epoch %d' % epoch)
            with Stats('Training Epoch Took'):
                results = model_trainer.process_one_epoch(train=True)

                print('Training results:')
                print(results)
                results_logger.log_metrics(results, epoch, train=True)

            logger.info('Saving the model')
            model.save_model('models/tmp.model')

            logger.info('Start Validation Epoch: Epoch %d' % epoch)
            with Stats('Validation Epoch Took'):
                results = model_trainer.process_one_epoch(train=False)

                print('Validation results:')
                print(results)

                results_logger.log_metrics(results, epoch, train=False)
            epoch += 1

    except:
        train_dr.stop_readers()
        test_dr.stop_readers()


if __name__ == '__main__':
    main()
