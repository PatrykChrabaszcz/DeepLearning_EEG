from src.data_reader import AnomalyDataReader, SequenceDataReader
from src.utils import Stats
import click

cache_path = '/home/chrabasp/data'


@click.command()
@click.option('--data_path', type=click.Path(exists=True), required=True)
@click.option('--sequence_size', default=500)
@click.option('--batch_size', default=128)
@click.option('--train_readers', default=6)
@click.option('--test_readers', default=6)
@click.option('--backend', default='Pytorch')
def main(data_path, sequence_size, batch_size, train_readers, test_readers, backend):

    # Initialize backend
    if backend.title() == 'Tensorflow':
        print('Will use Tensorflow backend')
        from src.tensorflow.model import SimpleRNN
        from src.tensorflow.model_trainer import ModelTrainer
    elif backend.title() == 'Pytorch':
        print('Will use PyTorch backend')
        from src.pytorch.model import SimpleRNN
        from src.pytorch.model_trainer import ModelTrainer
    else:
        raise NotImplementedError('Specify backend as Tensorflow or PyTorch')

    # Initialize data readers
    train_dr = AnomalyDataReader(data_path, readers_count=train_readers, batch_size=batch_size)
    test_dr = AnomalyDataReader(data_path, readers_count=test_readers, batch_size=batch_size,
                                data_type=SequenceDataReader.Validation_Data)

    model = SimpleRNN(22, 32, 2, 2)

    print('Number of parameters in the model %d' % model.count_params())
    model_trainer = ModelTrainer(model, train_dr, test_dr, sequence_size)

    try:
        while True:
            print('Train Epoch')
            with Stats('Epoch Took'):
                model_trainer.process_one_epoch(train=True)
            print('')

            print('Validation Epoch')
            with Stats('Validation Took'):
                model_trainer.process_one_epoch(train=False)
            print('')

    except InterruptedError:
        train_dr.stop_readers()
        test_dr.stop_readers()


if __name__ == '__main__':
    main()
