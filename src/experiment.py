from src.hpbandster.worker import Worker
from src.experiment_arguments import ExperimentArguments
from src.data_reading.data_reader import SequenceDataReader
import logging
import src.data_reading as reader_module
from src.utils import setup_logging
from src.hpbandster.bayesian_optimizer import BayesianOptimizer
from src.hpbandster.config_generator import ConfigGenerator
import src.hpbandster.budget_decoder
from src.dl_core.train_manager import TrainManager
import json


# Initialize logging
logger = logging.getLogger(__name__)


class Experiment:
    ExperimentBayesianOptimization = 'BayesianOptimization'
    ExperimentSingleTrain = 'ExperimentSingleTrain'
    ExperimentSingleEvaluation = 'ExperimentSingleEvaluation'
    ExperimentTypes = [ExperimentBayesianOptimization, ExperimentSingleTrain, ExperimentSingleEvaluation]

    @staticmethod
    # Parsing arguments
    def add_arguments(parser):
        parser.add_argument("--config_file", type=str, default='',
                            help="File with .pcs (parameter configuration space) data.")
        parser.add_argument("--model_class_name", default='SimpleRNN',
                            help="Model class used for the training.")
        parser.add_argument("--reader_class_name", default='AnomalyDataReader',
                            help="Reader class used for the training.")
        parser.add_argument("--budget_decoder_class_name", default='SimpleBudgetDecoder',
                            help="Class used to update setting based on higher budget.")
        parser.add_argument("--backend", default="Pytorch",
                            help="Whether to use Tensorflow or Pytorch.")
        parser.add_argument("--verbose", type=int, default=0, choices=[0, 1],
                            help="If set to 1 then log debug messages.")
        parser.add_argument("--is_master", type=int, default=0, choices=[0, 1],
                            help="If set to 1 then it will run thread for BO optimization.")
        parser.add_argument("--experiment_type", type=str, default=Experiment.ExperimentTypes[0],
                            choices=Experiment.ExperimentTypes,
                            help="TODO Write help message")
        parser.add_argument("--evaluation_log_dir", type=str, default='',
                            help="TODO Write help message")
        parser.add_argument("--evaluation_data_type", type=str, default=SequenceDataReader.Validation_Data,
                            choices=SequenceDataReader.DataTypes,
                            help="TODO Write help message")
        return parser

    def __init__(self):
        # Parse initial experiment arguments
        initial_arguments = ExperimentArguments(use_all_cli_args=False)
        initial_arguments.add_class_arguments(Experiment)
        initial_arguments.get_arguments(sections=('experiment',))

        verbose = initial_arguments.verbose
        setup_logging(logging.DEBUG if verbose else logging.INFO)

        self.is_master = initial_arguments.is_master
        self.experiment_type = initial_arguments.experiment_type
        self.evaluation_data_type = initial_arguments.evaluation_data_type
        self.evaluation_log_dir = initial_arguments.evaluation_log_dir

        backend = initial_arguments.backend.title()
        # Initialize backend (Currently only PyTorch implemented)
        if backend == 'Tensorflow':
            logger.warning('Tensorflow backend is outdated.')
            import src.dl_tensorflow.model as model_module
            from src.dl_tensorflow.model_trainer import ModelTrainer as TrainerClass
        elif backend == 'Pytorch':
            logger.info('Will use PyTorch backend.')
            import src.dl_pytorch.model as model_module
            from src.dl_pytorch.model_trainer import ModelTrainer as TrainerClass
        else:
            raise NotImplementedError('Specify backend as Tensorflow or PyTorch')

        self.ModelClass = getattr(model_module, initial_arguments.model_class_name)
        self.ReaderClass = getattr(reader_module, initial_arguments.reader_class_name)
        self.BudgetDecoderClass = getattr(src.hpbandster.budget_decoder, initial_arguments.budget_decoder_class_name)
        self.TrainerClass = TrainerClass

        # Populate experiment arguments with arguments from specific classes
        self.experiment_arguments = ExperimentArguments(use_all_cli_args=True)
        self.experiment_arguments.add_class_arguments(Experiment)
        self.experiment_arguments.add_class_arguments(self.ModelClass)
        self.experiment_arguments.add_class_arguments(self.ReaderClass)
        self.experiment_arguments.add_class_arguments(self.TrainerClass)
        self.experiment_arguments.add_class_arguments(TrainManager)
        self.experiment_arguments.add_class_arguments(BayesianOptimizer)
        self.experiment_arguments.add_class_arguments(ConfigGenerator)
        self.experiment_arguments.get_arguments()

        self.train_manager = TrainManager(ModelClass=self.ModelClass, ReaderClass=self.ReaderClass,
                                          TrainerClass=self.TrainerClass,
                                          **self.experiment_arguments.get_arguments())
        self.budget_decoder = self.BudgetDecoderClass(**self.experiment_arguments.get_arguments())
        self.worker = None

        try:
            self.config_space = ExperimentArguments.read_configuration_space(initial_arguments.config_file)
        except Exception:
            logger.warning('Could not find config_space file. '
                           'It will not be possible to run hyperparameter optimization.')

    def run_bayesian_optimization(self):
        """
        If is_master is 1 then runs Hyper-Parameter Search optimizer in a separate thread

        Runs one worker that will read configurations generated by Hyper-Parameter Search optimizer
        and report results
        :return:
        """
        if self.is_master:
            # Manage conf file creation and removal
            with BayesianOptimizer(self.config_space, **self.experiment_arguments.get_arguments()) as optimizer:
                self.start_worker()
                try:
                    logger.info('Starting Master')
                    optimizer.run()
                except (Exception, KeyboardInterrupt):
                    logger.info('Shutting down parallel workers.')
                    optimizer.shutdown()
        else:
            self.start_worker()

    def start_worker(self):
        logger.info('Starting Worker')

        self.worker = Worker(train_manager=self.train_manager, budget_decoder=self.budget_decoder,
                             experiment_args=self.experiment_arguments, **self.experiment_arguments.get_arguments())
        self.worker.run(background=True if self.is_master else False)

    def run_single_train(self):
        self.train_manager.init_new_log_dir()

        train_metrics = self.train_manager.train(self.experiment_arguments)
        valid_metrics = self.train_manager.validate(self.experiment_arguments, log_dir=self.train_manager.log_dir,
                                                    data_type=SequenceDataReader.Validation_Data)

        logger.info('Train Metrics:')
        logger.info(json.dumps(train_metrics.get_summarized_results(), indent=2, sort_keys=True))
        logger.info('Validation Metrics:')
        logger.info(json.dumps(valid_metrics.get_summarized_results(), indent=2, sort_keys=True))

    def run_single_evaluation(self):
        if self.evaluation_data_type == SequenceDataReader.Train_Data:
            if self.experiment_arguments.random_mode != 0:
                logger.warning('Evaluation of training data but random mode is non 0 (Are you sure?)')
            if self.experiment_arguments.continuous != 0:
                logger.warning('Evaluation of training data but continuous is not 0 (Are you sure?)')
            if self.experiment_arguments.forget_state == 1:
                logger.warning('Evaluation of training data but forget_state is set to 1 (Are you sure?)')
            if self.experiment_arguments.balanced == 1:
                logger.warning('Evaluation of training data but balanced is set to 1 (Are you sure?)')

        metrics = self.train_manager.validate(self.experiment_arguments, log_dir=self.evaluation_log_dir,
                                              data_type=self.evaluation_data_type)

        logger.info('Metrics %s:' % self.evaluation_data_type)
        logger.info(json.dumps(metrics.get_summarized_results(), indent=2, sort_keys=True))

    def main(self):
        func_dic = {
            self.ExperimentBayesianOptimization: self.run_bayesian_optimization,
            self.ExperimentSingleTrain: self.run_single_train,
            self.ExperimentSingleEvaluation: self.run_single_evaluation,
        }
        logger.info('Main Function To Execute: %s' % self.experiment_type)
        func_dic[self.experiment_type]()

