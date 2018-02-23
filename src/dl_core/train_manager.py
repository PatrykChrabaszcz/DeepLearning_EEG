from src.data_reading import SequenceDataReader
from datetime import datetime
import logging
import os


# Initialize logging
logger = logging.getLogger(__name__)


# Connects all components: Model, Reader, Trainer and runs on training/validation/test data.
# Used for single experiments as well as distributed experiments with Hyperparameter Optimization.
class TrainManager:
    def __init__(self, ModelClass, ReaderClass, TrainerClass, save_training_logs, working_dir, **kwargs):
        self.ModelClass = ModelClass
        self.ReaderClass = ReaderClass
        self.TrainerClass = TrainerClass
        self.save_training_logs = save_training_logs
        self.working_dir = working_dir
        self.log_dir = None
        self.init_new_log_dir()

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--save_training_logs", type=int, default=0,
                            help="Whether to save the model and results from the training.")

    def init_new_log_dir(self):
        self.log_dir = os.path.join(self.working_dir, 'training_logs',
                                    datetime.utcnow().strftime('%Y_%m_%d__%H_%M_%S_%f'))
        os.makedirs(self.log_dir, exist_ok=False)

    def new_model(self, experiment_arguments):
        args = experiment_arguments.get_arguments()
        context_size = self.ReaderClass.context_size(**args)
        input_size = self.ReaderClass.input_size(**args)
        output_size = self.ReaderClass.output_size(**args)

        model = self.ModelClass(input_size=input_size, output_size=output_size,
                                context_size=context_size, **args)

        logger.info('Number of parameters in the model %d' % model.count_params())
        return model

    def train(self, experiment_arguments):
        model = self.new_model(experiment_arguments)

        train_metrics = self._run(model, experiment_arguments, SequenceDataReader.Train_Data, train=True)

        if self.save_training_logs:
            experiment_arguments_file = os.path.join(self.log_dir, 'config.ini')
            experiment_arguments.save_to_file(file_path=experiment_arguments_file)
            train_metrics.save(directory=self.log_dir)
            model.save_model(os.path.join(self.log_dir, 'model'))

        return train_metrics

    def validate(self, experiment_arguments, log_dir, data_type=SequenceDataReader.Test_Data):
        model = self.new_model(experiment_arguments)
        model_file_path = os.path.join(log_dir, 'model')
        model.load_model(model_file_path)
        logger.info('Model loaded from %s' % model_file_path)

        test_metrics = self._run(model, experiment_arguments, data_type, train=False)

        if self.save_training_logs:
            test_metrics.save(os.path.join(log_dir, 'additional'))

        return test_metrics

    def _run(self, model, experiment_arguments, data_type=SequenceDataReader.Train_Data, train=False):
        args = experiment_arguments.get_arguments()

        if data_type is not SequenceDataReader.Train_Data and train is True:
            raise RuntimeError('You try to train the network using validation or test data!')

        if data_type is SequenceDataReader.Train_Data and train is False:
            logger.warning('You use training data but you do not train the network.')

        offset_size = model.offset_size(sequence_size=args['initial_sequence_size'])
        logger.info('Data reader will use an offset:  %d' % offset_size)

        allow_smaller_batch = False if train else True
        dr = self.ReaderClass(offset_size=offset_size, allow_smaller_batch=allow_smaller_batch,
                              state_initializer=model.initial_state, data_type=data_type, **args)

        trainer = self.TrainerClass(model=model, **args)

        metrics = trainer.run(data_reader=dr, train=train)

        return metrics
