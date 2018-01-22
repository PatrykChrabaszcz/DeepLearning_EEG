import logging
import re
from os import makedirs

import click

from src.data_reading.data_reader import AnomalyDataReader
from src.data_reading.data_reader import SequenceDataReader
from src.dl_pytorch.model_trainer import ModelTrainer
from src.utils import setup_logging

# Initialize logging
logger = logging.getLogger(__name__)
setup_logging()


@click.command()
@click.option('--data_path', type=click.Path(exists=True), required=True)
@click.option('--model_path', type=click.Path(exists=True), required=True)
@click.option('--sequence_size', default=500)
@click.option('--batch_size', default=16)
@click.option('--readers_count', default=2)
@click.option('--limit_duration', default=None)
@click.option('--limit_examples', default=None)
def main(data_path, model_path, sequence_size, batch_size, readers_count, limit_duration, limit_examples):

    if limit_examples is not None:
        limit_examples = int(limit_examples)
    if limit_duration is not None:
        limit_duration = int(limit_duration)

    from src.dl_pytorch.model import SimpleRNN
    logger.info('Will use PyTorch backend')

    [hidden_size] = re.findall('HiddenSize\((\d+)\)', model_path)
    [num_layers] = re.findall('Layers\((\d+)\)', model_path)
    [cell_type] = re.findall('Cell\((\w+)\)', model_path)
    [label_type] = re.findall('LabelType\((\w+)\)', model_path)
    [use_context] = re.findall('Context\((\w+)\)', model_path)
    context_size = AnomalyDataReader.context_sizes[label_type]
    model = SimpleRNN(input_size=22, hidden_size=int(hidden_size), num_layers=int(num_layers), output_size=2,
                      dropout=0.0, cell_type=cell_type, use_context=use_context=='True', context_size=context_size)
    model.load_model(model_path)

    # Initialize data readers
    dr_kwargs = {
        'cache_path': data_path,
        'label_type': label_type,
        'limit_examples': limit_examples,
        'limit_duration': limit_duration,
        'batch_size': batch_size,
        'readers_count': readers_count,
        'allow_smaller_batch': True,
        'state_initializer': model.initial_state
    }
    train_dr = AnomalyDataReader(data_type=SequenceDataReader.Train_Data, **dr_kwargs)
    valid_dr = AnomalyDataReader(data_type=SequenceDataReader.Validation_Data, **dr_kwargs)
    test_dr = AnomalyDataReader(data_type=SequenceDataReader.Test_Data, **dr_kwargs)

    model_trainer = ModelTrainer(model, 0.0, 0.0, sequence_size, loss_type='classification_all')

    logger.info('Number of parameters in the model %d' % model.count_params())

    for (name, dr) in [('Train', train_dr), ('Validation', valid_dr), ('Test', test_dr)]:
        dr.start_readers()

        metrics = model_trainer.process_one_epoch(forget_state=False, sequence_size=sequence_size, data_reader=dr,
                                                  randomize=False, update=False, iterations=None)

        print('Results for %s' % name)

        print(metrics.get_summarized_results())
        makedirs('results', exist_ok=True)
        metrics.save_detailed_output('results/%s.json'%name)
        dr.stop_readers()

if __name__ == '__main__':
    main()
