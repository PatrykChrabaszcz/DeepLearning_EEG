import logging
import os
from time import strftime

import click

from src.data_reading.data_reader import SequenceDataReader
from src.dl_core.model import ModelBase
from src.dl_core.model_trainer import ModelTrainerBase
from src.result_logger import ResultsLogger
from src.utils import setup_logging
from test_one_label.data_reader import OneLabelDataReader

# Initialize logging
logger = logging.getLogger(__name__)
setup_logging()


@click.command()
@click.option('--sequence_size', default=500)
@click.option('--batch_size', default=64)
@click.option('--learning_rate', default=0.001)
@click.option('--weight_decay', default=0.0)
@click.option('--readers', default=2)
@click.option('--num_layers', default=3)
@click.option('--num_neurons', default=128)
@click.option('--backend', default='Pytorch')
@click.option('--forget_state', is_flag=True)
@click.option('--cell_type', type=click.Choice(ModelBase.cell_types), default=ModelBase.cell_types[0])
@click.option('--loss_type', type=click.Choice(ModelTrainerBase.loss_types), default=ModelTrainerBase.loss_types[0])
def main(sequence_size, batch_size, learning_rate, weight_decay, readers,
         num_layers, num_neurons, backend, forget_state, cell_type, loss_type):
    print(os.getcwd())
    logger.info('Will use PyTorch backend')
    from src.dl_pytorch.model import SimpleRNN
    from src.dl_pytorch.model_trainer import ModelTrainer

    time_string = strftime("%Y%m%d-%H%M%S")
    experiment_name = 'Random_Test_Backend(%s)_SeqSize(%s)_LR(%s)_WD(%s)_%s' % \
                      (backend, sequence_size, learning_rate, weight_decay, time_string)

    logger.info('Experiment name: %s' % experiment_name)

    model = SimpleRNN(22, num_neurons, num_layers, 2, cell_type)

    # Initialize data readers
    train_dr = OneLabelDataReader(readers_count=readers, batch_size=batch_size, state_initializer=model.initial_state,
                                  data_type=SequenceDataReader.Train_Data)
    test_dr = OneLabelDataReader(readers_count=readers, batch_size=batch_size, state_initializer=model.initial_state,
                                 data_type=SequenceDataReader.Train_Data)

    model_trainer = ModelTrainer(model, learning_rate, weight_decay, train_dr, test_dr, loss_type)
    results_logger = ResultsLogger(experiment_name=experiment_name)

    logger.info('Number of parameters in the model %d' % model.count_params())
    epoch = 1
    try:
        import time
        while True:
            logger.info('\nStart Train Epoch (ForgetState %s): Epoch %d\n' % (forget_state, epoch))
            results = model_trainer.process_one_epoch(forget_state=forget_state, sequence_size=sequence_size,
                                                      randomize=True, train=True, update=True)
            logger.info('Training results (online):')
            print(results)

            logger.info('\nCompute results for Train/Validation splits: Epoch %d\n' % epoch)

            results = model_trainer.process_one_epoch(forget_state=True, sequence_size=sequence_size,
                                                      train=True, update=False)
            logger.info('Training results (forget):')
            print(results)
            results_logger.log_metrics(results, epoch, forget_state=True, train=True)

            results = model_trainer.process_one_epoch(forget_state=False, sequence_size=sequence_size,
                                                      train=True, update=False)
            logger.info('Training results (remember):')
            print(results)
            results_logger.log_metrics(results, epoch, forget_state=False, train=True)

            logger.info('Saving the model')
            model.save_model('models/%s.model' % experiment_name)

            logger.info('Validation results (forget)')
            results = model_trainer.process_one_epoch(forget_state=True, sequence_size=sequence_size,
                                                      train=False, update=False)
            print(results)
            results_logger.log_metrics(results, epoch, forget_state=True, train=False)

            logger.info('Validation results (remember)')
            results = model_trainer.process_one_epoch(forget_state=False, sequence_size=sequence_size,
                                                      train=False, update=False)
            print(results)
            print("\n\n\n\n")
            results_logger.log_metrics(results, epoch, forget_state=False, train=False)

            epoch += 1

    except Exception as e:
        train_dr.stop_readers()
        test_dr.stop_readers()
        raise e


if __name__ == '__main__':
    main()
