import logging
import re
from os import makedirs
import os
import click
import json

from src.data_reading.anomaly_data_reader import AnomalyDataReader
from src.data_reading.data_reader import SequenceDataReader
from src.dl_pytorch.model_trainer import ModelTrainer
from src.utils import setup_logging

# Initialize logging
logger = logging.getLogger(__name__)
setup_logging()


@click.command()
@click.option('--data_path', type=click.Path(exists=True), required=True)
@click.option('--model_path', type=click.Path(exists=True), required=True)
@click.option('--log_path', type=click.Path(exists=True), required=True)
@click.option('--sequence_size', default=500)
@click.option('--batch_size', default=16)
@click.option('--readers_count', default=2)
@click.option('--limit_duration', default=None)
@click.option('--limit_examples', default=None)
def main(data_path, model_path, log_path, sequence_size, batch_size, readers_count, limit_duration, limit_examples):

    if limit_examples is not None:
        limit_examples = int(limit_examples)
    if limit_duration is not None:
        limit_duration = int(limit_duration)

    from src.dl_pytorch.model import SimpleRNN
    logger.info('Will use PyTorch backend')

    with open(os.path.join(log_path, 'model_args.json')) as f:
        model_args = json.load(f)
    with open(os.path.join(log_path, 'trainer_args.json')) as f:
        trainer_args = json.load(f)
        trainer_args["loss_type"] = "classification_all"
    with open(os.path.join(log_path, 'reader_args.json')) as f:
        reader_args = json.load(f)

    context_size = AnomalyDataReader.context_size(**reader_args)
    input_size = AnomalyDataReader.input_size(**reader_args)
    output_size = AnomalyDataReader.output_size(**reader_args)

    model = SimpleRNN(input_size=input_size, output_size=output_size, context_size=context_size, **model_args)
    model.load_model(model_path)

    # Initialize data readers
    offset_size = model.offset_size()
    train_dr = AnomalyDataReader(offset_size=offset_size, allow_smaller_batch=True, continuous=False,
                                 state_initializer=model.initial_state, data_type=AnomalyDataReader.Train_Data,
                                 **reader_args)
    valid_dr = AnomalyDataReader(offset_size=offset_size, allow_smaller_batch=True, continuous=False,
                                 state_initializer=model.initial_state, data_type=AnomalyDataReader.Validation_Data,
                                 **reader_args)
    test_dr = AnomalyDataReader(offset_size=offset_size, allow_smaller_batch=True, continuous=False,
                                state_initializer=model.initial_state, data_type=AnomalyDataReader.Test_Data,
                                **reader_args)

    model_trainer = ModelTrainer(model=model, **trainer_args)

    logger.info('Number of parameters in the model %d' % model.count_params())

    for (name, dr) in [('Train', train_dr), ('Validation', valid_dr), ('Test', test_dr)]:
        dr.start_readers()

        metrics = model_trainer.process_one_epoch(forget_state=False, sequence_size=sequence_size, data_reader=dr,
                                                  randomize=False, update=False, iterations=None)

        print('Results for %s' % name)

        makedirs('results', exist_ok=True)
        metrics.save_detailed_output('results/%s.json' % name)
        dr.stop_readers()

if __name__ == '__main__':
    main()
