from src.data_reader import AnomalyDataReader, SequenceDataReader
from src.core.metrics import ClassificationMetrics
from src.utils import Stats, setup_logging
import click
import logging
from torch.autograd import Variable
import torch
from src.data_reader import SequenceDataReader
import numpy as np
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
            #self.inputs = []

        def append(self, output, labels, inputs):
            assert (len(output) == len(labels))
            prob = [np.exp(o[lab]) / (np.exp(o[0]) + np.exp(o[1])) for (o, lab) in zip(output, labels)]
            labels = labels.flatten().tolist()

            #self.inputs.append(inputs)
            self.output.extend(prob)
            self.labels.extend(labels)

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

        for id, result in self.results.items():
            plt.title(str(id))
            # inputs = np.concatenate(result.inputs)
            #sns.tsplot(result.output, color='blue', alpha=0.5)
            #sns.tsplot(result.labels, color='red', alpha=0.5)

            f, axarr = plt.subplots(23, sharex=True)
            # for i in range(22):
            #     sns.tsplot(inputs[:, i], color='green', ax=axarr[i])

            sns.tsplot(result.output, color='red', ax=axarr[22])

            plt.show()


@click.command()
@click.option('--data_path', type=click.Path(exists=True), required=True)
@click.option('--sequence_size', default=500)
@click.option('--batch_size', default=16)
@click.option('--train_readers', default=5)
@click.option('--test_readers', default=5)
@click.option('--limit_duration', default=500)
@click.option('--limit_examples', default=None)
@click.option('--forget_state', is_flag=True)
def main(data_path, sequence_size, batch_size, train_readers, test_readers, limit_duration, limit_examples,
         forget_state):
    from src.pytorch.model import SimpleRNN
    logger.info('Will use PyTorch backend')

    model = SimpleRNN(22, 32, 1, 2)
    model.load_model('models/tmp.model')
    metrics = ClassificationMetrics()
    output_aggregator = OutputAggregator()
    # Initialize data readers
    train_dr = AnomalyDataReader(data_path, limit_examples, limit_duration,
                                 readers_count=train_readers, batch_size=batch_size,
                                 state_initializer=model.initial_state)
    test_dr = AnomalyDataReader(data_path, limit_examples, limit_duration,
                                readers_count=test_readers, batch_size=batch_size,
                                state_initializer=model.initial_state, data_type=SequenceDataReader.Validation_Data)

    logger.info('Number of parameters in the model %d' % model.count_params())

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
            hidden = model.import_state(dr.get_states(ids, forget=forget_state), cuda=False)
            outputs, hidden = model(batch, hidden)
            metrics.append_results(ids, outputs.data.numpy(), labels.data.numpy(), train=train)
            output_aggregator.append_results(ids, outputs.data.numpy(), labels.data.numpy(), batch.data.numpy())

            dr.set_states(ids, model.export_state(hidden))

            iteration += 1

    except SequenceDataReader.EpochDone:
        print('%d Iterations in this epoch' % iteration)

        print(metrics.finish_epoch(train))

    logger.info('Evaluation finished')
    dr.stop_readers()
    #output_aggregator.plot()

if __name__ == '__main__':
    main()
