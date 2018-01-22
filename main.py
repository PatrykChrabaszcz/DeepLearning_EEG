import logging
from os import path, makedirs
from time import strftime
import argparse
import numpy as np
import src.data_reading as reader_module
from src.result_logger import ResultsLogger
from src.utils import setup_logging
import configparser

# Initialize logging
logger = logging.getLogger(__name__)


def add_arguments(parser):
    parser.add_argument("-m", "--model_class", help="Model class used for the training.", default='SimpleRNN')
    parser.add_argument("-r", "--reader_class", help="Reader class used for the training.", default='AnomalyDataReader')
    parser.add_argument("-c", "--conf_file", type=str, help="Model configuration file.", default="")
    parser.add_argument("-b", "--backend", help="Whether to use Tensorflow or Pytorch.", default="Pytorch")
    parser.add_argument("-v", "--verbose", type=int, default=0, choices=[0, 1],
                        help="If set to 1 then log debug messages")


def extract_arguments(conf_file, unknown_args, ModelClass, ModelTrainer, ReaderClass):
    # Extract parameters for the model
    model_parser = argparse.ArgumentParser()
    ModelClass.add_arguments(model_parser)

    # Extract parameters for the trainer
    trainer_parser = argparse.ArgumentParser()
    ModelTrainer.add_arguments(trainer_parser)

    # Extract parameters for the reader
    reader_parser = argparse.ArgumentParser()
    ReaderClass.add_arguments(reader_parser)

    if conf_file != "":
        conf = configparser.ConfigParser()
        conf.read(conf_file)

        for parser, name in ((model_parser, "model"), (trainer_parser, "trainer"), (reader_parser, "reader")):
            parser.set_defaults(**dict(conf.items(name)))

    res = []
    for parser in [model_parser, trainer_parser, reader_parser]:
        args, unknown_args = parser.parse_known_args(args=unknown_args)
        res.append(vars(args))

    if len(unknown_args) > 0:
        logger.error('There are some unknown arguments provided by the user')
        logger.error(unknown_args)
        raise RuntimeError('Unknown CLI arguments')

    return res


def main():
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args, unknown_args = parser.parse_known_args()

    log_level = logging.INFO if args.verbose == 0 else logging.DEBUG
    setup_logging(log_level)

    # Initialize backend
    if args.backend.title() == 'Tensorflow':
        # Tensorflow outdated
        logger.info('Will use Tensorflow backend')
        import src.dl_tensorflow.model as model_module
        from src.dl_tensorflow.model_trainer import ModelTrainer
    elif args.backend.title() == 'Pytorch':
        logger.info('Will use PyTorch backend')
        import src.dl_pytorch.model as model_module
        from src.dl_pytorch.model_trainer import ModelTrainer
    else:
        raise NotImplementedError('Specify backend as Tensorflow or PyTorch')

    # Create model and reader classes from user provided arguments
    ModelClass = getattr(model_module, args.model_class)
    ReaderClass = getattr(reader_module, args.reader_class)

    model_args, trainer_args, reader_args = extract_arguments(args.conf_file, unknown_args,
                                                              ModelClass, ModelTrainer, ReaderClass)

    logger.debug(args)
    logger.debug(model_args)
    logger.debug(reader_args)
    context_size = ReaderClass.context_size(**reader_args)
    input_size = ReaderClass.input_size(**reader_args)
    output_size = ReaderClass.output_size(**reader_args)

    model = ModelClass(input_size=input_size, output_size=output_size, context_size=context_size, **model_args)

    logger.info('Printing model structure (Num of parameters %d)' % model.count_params())

    # Initialize data readers
    offset_size = model.offset_size()
    logger.info('Data readers will use an offset:  %d' % offset_size)

    continuous = trainer_args['iterations_per_epoch'] > 0
    train_dr = ReaderClass(offset_size=offset_size, allow_smaller_batch=False, continuous=continuous,
                           state_initializer=model.initial_state, data_type=ReaderClass.Train_Data, **reader_args)

    reader_args['balanced'] = False
    valid_dr = ReaderClass(offset_size=offset_size, allow_smaller_batch=True, continuous=False,
                           state_initializer=model.initial_state, data_type=ReaderClass.Validation_Data, **reader_args)

    model_trainer = ModelTrainer(model=model, **trainer_args)

    experiment_name = 'Model(%s)_Reader(%s)_Time(%s)' % (args.model_class, args.reader_class, strftime("%Y%m%d-%H%M%S"))
    logger.info('Experiment name: %s' % experiment_name)

    results_logger = ResultsLogger(reader_args, model_args, trainer_args, experiment_name=experiment_name)

    epoch = 1
    best_validation_loss = np.inf
    try:
        train_dr.start_readers()
        valid_dr.start_readers()
        while True:
            logger.info('\nStart Train Epoch (ForgetState %s): Epoch %d\n' % (reader_args['forget_state'], epoch))
            iterations = None if trainer_args['iterations_per_epoch'] <= 0 else trainer_args['iterations_per_epoch']
            metrics = model_trainer.process_one_epoch(forget_state=reader_args['forget_state'],
                                                      sequence_size=reader_args['sequence_size'],
                                                      data_reader=train_dr, randomize=True, update=True,
                                                      iterations=iterations)
            results = metrics.get_summarized_results()
            logger.info('Training results (online):')
            print(results)
            results_logger.log_metrics(results, epoch, forget_state=reader_args['forget_state'], train=True)

            metrics = model_trainer.process_one_epoch(forget_state=False,
                                                      sequence_size=reader_args['sequence_size'],
                                                      data_reader=valid_dr, randomize=False, update=False)
            results = metrics.get_summarized_results()
            logger.info('Validation results (remember)')
            print(results)
            print("\n\n\n\n")
            results_logger.log_metrics(results, epoch, forget_state=False, train=False)

            if results['loss'] == np.nan:
                logger.warn('Validation Loss is Nan, throw an exception...')
                raise RuntimeError('Validation error is Nan')

            if results['loss'] < best_validation_loss:
                best_validation_loss = results['loss']

                logger.info('Saving current best model (Loss %s)' % best_validation_loss)
                makedirs('models', exist_ok=True)
                model.save_model(path.join('models', experiment_name))

            epoch += 1

    except Exception as e:
        logger.warn('Exception detected, will try to stop the readers before shutting down...')
        train_dr.stop_readers()
        valid_dr.stop_readers()
        raise e


if __name__ == '__main__':
    main()

