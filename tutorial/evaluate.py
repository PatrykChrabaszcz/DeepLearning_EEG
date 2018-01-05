from src.data_reader import SequenceDataReader
from src.core.metrics import RegressionMetrics
from tutorial.data_reader import TutorialDataReader
from tutorial.model import TutorialRNN
from tutorial.model_trainer import ModelTrainer
from src.utils import Stats, setup_logging
import click
import logging
from src.result_logger import ResultsLogger
from torch.autograd import Variable
import torch
from src.data_reader import SequenceDataReader

from time import strftime

# Initialize logging
logger = logging.getLogger(__name__)
setup_logging()


# Class used to aggregate outputs from the network
class OutputAggregator:
    class Example:
        def __init__(self, example_id):
            self.example_id = example_id
            self.output = []
            self.labels = []
            self.inputs = []

        def append(self, output, labels, inputs):
            output = output.flatten().tolist()
            labels = labels.flatten().tolist()
            inputs = inputs.flatten().tolist()

            self.output.extend(output)
            self.labels.extend(labels)
            self.inputs.extend(inputs)

    def __init__(self):
        self.results = {}

    def append_results(self, ids, output, labels, inputs):
        for example_id, o, l, i in zip(ids, output, labels, inputs):
            if example_id not in self.results:
                self.results[example_id] = OutputAggregator.Example(example_id)

            self.results[example_id].append(o, l, i)

    def plot(self):
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.tsplot(self.results[0].output, color='blue', alpha=0.5)
        sns.tsplot(self.results[0].labels, color='red', alpha=0.5)
        sns.tsplot(self.results[0].inputs, color='green', alpha=0.5)

        plt.show()


@click.command()
@click.option('--sequence_size', default=100)
@click.option('--batch_size', default=64)
@click.option('--learning_rate', default=0.001)
@click.option('--train_readers', default=5)
@click.option('--test_readers', default=5)
@click.option('--forget_state', is_flag=True)
def main(sequence_size, batch_size, learning_rate, train_readers, test_readers, forget_state):

    model = TutorialRNN(1, 32, 1, 1)
    model.load_model('models/tmp.model')
    # Initialize data readers
    train_dr = TutorialDataReader(readers_count=train_readers, batch_size=batch_size,
                                  state_initializer=model.initial_state)
    test_dr = TutorialDataReader(readers_count=test_readers, batch_size=batch_size,
                                 state_initializer=model.initial_state, data_type=SequenceDataReader.Validation_Data)

    logger.info('Number of parameters in the model %d' % model.count_params())

    metrics = RegressionMetrics()
    output_aggregator = OutputAggregator()

    iteration = 0

    train = True
    dr = train_dr if train else test_dr
    dr.initialize_epoch(randomize=False, sequence_size=sequence_size)
    dr.start_readers()

    try:
        while True:
            batch, time, labels, ids = dr.get_batch()
            batch = Variable(torch.from_numpy(batch))
            labels = Variable(torch.from_numpy(labels))
            hidden = model.import_state(dr.get_states(ids, forget=forget_state))
            outputs, hidden = model(batch, hidden)
            metrics.append_results(ids, outputs.data.numpy(), labels.data.numpy(), train=train)
            output_aggregator.append_results(ids, outputs.data.numpy(), labels.data.numpy(), batch.data.numpy())

            dr.set_states(ids, model.export_state(hidden))

            iteration += 1

    except SequenceDataReader.EpochDone:
        print('%d Iterations in this epoch' % iteration)

    result = metrics.finish_epoch(train=train)
    output_aggregator.plot()
    dr.stop_readers()

    return result


if __name__ == '__main__':
    main()
