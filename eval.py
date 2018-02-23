import logging
from src.experiment import Experiment
import os
import json
from datetime import datetime
# Initialize logging
logger = logging.getLogger(__name__)


class TrainManager:
    def __init__(self, ModelClass, ReaderClass, TrainerClass, save_training_logs, working_dir, **kwargs):
        self.ModelClass = ModelClass
        self.ReaderClass = ReaderClass
        self.TrainerClass = TrainerClass
        self.save_training_logs = save_training_logs
        self.log_dir = os.path.join(working_dir, 'training_logs',
                                    datetime.utcnow().strftime('%Y_%m_%d__%H_%M_%S_%f'))

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--save_training_logs", type=int, default=0,
                            help="Whether to save the model and results from the training.")

    def train_and_validate(self, experiment_arguments):
        args = experiment_arguments.get_arguments()

        context_size = self.ReaderClass.context_size(**args)
        input_size = self.ReaderClass.input_size(**args)
        output_size = self.ReaderClass.output_size(**args)

        model = self.ModelClass(input_size=input_size, output_size=output_size, context_size=context_size,
                                **args)

        offset_size = model.offset_size(sequence_size=args['initial_sequence_size'])
        logger.info('Data readers will use an offset:  %d' % offset_size)

        train_dr = self.ReaderClass(offset_size=offset_size, allow_smaller_batch=False,
                                    state_initializer=model.initial_state,
                                    data_type=self.ReaderClass.Train_Data,
                                    **args)

        valid_dr = self.ReaderClass(offset_size=offset_size, allow_smaller_batch=True,
                                    state_initializer=model.initial_state,
                                    data_type=self.ReaderClass.Validation_Data,
                                    **args)

        trainer = self.TrainerClass(model=model, **args)

        # Train and validate the model
        train_metrics = trainer.run(data_reader=train_dr, train=True)
        evaluation_metrics = trainer.run(data_reader=valid_dr, train=False)

        if self.save_training_logs:
            os.makedirs(self.log_dir, exist_ok=False)

            with open(os.path.join(self.log_dir, 'arguments.json'), 'w') as f:
                json.dump(args, f, sort_keys=True, indent=2)

            train_metrics.save(os.path.join(self.log_dir, 'train'))
            evaluation_metrics.save(os.path.join(self.log_dir, 'valid'))

            model.save_model(os.path.join(self.log_dir, 'model'))

        return train_metrics, evaluation_metrics


if __name__ == '__main__':
    logger.info('Create experiment')
    experiment = Experiment()

    logger.info('Start hyper-parameter search')
    experiment.run_search_experiment()
