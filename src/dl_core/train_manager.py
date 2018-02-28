from src.data_reading import SequenceDataReader
from datetime import datetime
import logging
import os


# Initialize logging
logger = logging.getLogger(__name__)


# Connects all components: Model, Reader, Trainer and runs on training/validation/test data.
# Used for single experiments as well as distributed experiments with Hyperparameter Optimization.
class TrainManager:
    def __init__(self, ModelClass, ReaderClass, TrainerClass, main_logs_dir, run_log_dir, **kwargs):
        self.ModelClass = ModelClass
        self.ReaderClass = ReaderClass
        self.TrainerClass = TrainerClass
        self.main_logs_dir = main_logs_dir
        self.run_log_dir = run_log_dir
        self.log_dir = os.path.join(main_logs_dir, run_log_dir)

    @staticmethod
    def add_arguments(parser):
        parser.section('train_manager')
        parser.add_argument("main_logs_dir", type=str,
                            help="Main directory for training logs from different runs. "
                                 "New run will be logged inside run_log_dir or inside a new directory if run_log_dir"
                                 "is left empty.")
        parser.add_argument("run_log_dir", type=str, default="",
                            help="Directory for logs for the current run. If trained model already present then will "
                                 "use it.")

    def new_log_dir(self):
        self.run_log_dir = datetime.utcnow().strftime('%Y_%m_%d__%H_%M_%S_%f')
        self.log_dir = os.path.join(self.main_logs_dir, self.run_log_dir)
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

        if self.run_log_dir is "":
            self.new_log_dir()

        # Try to restore the model
        model_path = os.path.join(self.log_dir, 'model')
        try:
            model.load_model(model_path)
            logger.info('Model loaded from %s' % model_path)
        except FileNotFoundError:
            logger.info('Could not load the model, %s does not exist.' % model_path)

        train_metrics = self._run(model, experiment_arguments, SequenceDataReader.Train_Data, train=True)

        # Save training configuration, training logs and model
        experiment_arguments.run_log_dir = self.run_log_dir
        experiment_arguments_file = os.path.join(self.log_dir, 'config.ini')
        experiment_arguments.save_to_file(file_path=experiment_arguments_file)
        train_metrics.save(directory=self.log_dir)
        model.save_model(model_path)

        return train_metrics

    def validate(self, experiment_arguments, data_type=SequenceDataReader.Test_Data):
        model = self.new_model(experiment_arguments)
        model_file_path = os.path.join(self.log_dir, 'model')
        model.load_model(model_file_path)
        logger.info('Model loaded from %s' % model_file_path)

        test_metrics = self._run(model, experiment_arguments, data_type, train=False)

        # Save validation/test results in a separate folder (we might want to store multiple validation results)
        test_metrics.save(os.path.join(self.log_dir, datetime.utcnow().strftime('%Y_%m_%d__%H_%M_%S_%f')))

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
