from src.data_reading.data_reader import SequenceDataReader
import os
import numpy as np
from src.utils import nested_list_len
import logging
import gzip
import pickle

logger = logging.getLogger(__name__)


# This does not use multi-threading at all
class MnistDataReader(SequenceDataReader):
    Mnist_Seq_Size = 784

    # We fix offset_size at 0 because we do not consider a situation when one example (size 784) will be split into
    # chunks and processed using CNN.
    # We fix random_mode to 0 because we do not see the point of starting from a random position within the sequence.
    class ExampleInfo(SequenceDataReader.SequenceExampleInfo):
        def __init__(self, example_id, data, label):
            super().__init__(example_id=example_id, offset_size=0, random_mode=0)
            # This assert is also fulfilled when limit_duration is None
            self.data = data
            self.label = label

            # Nothing that comes to my mind and can be used as context for this dataset
            self.context = None

        def get_length(self):
            return MnistDataReader.Mnist_Seq_Size

        def read_data(self, serialized):
            index, sequence_size = serialized
            assert index == 0, 'For Mnist dataset only reading from the beginning is considered.'

            data = self.data[:sequence_size]
            # data = (data - self.mean) / self.std
            # Time might be needed for some advanced models like PhasedLSTM
            time = np.reshape(np.arange(0, sequence_size), newshape=[sequence_size, 1])

            label = np.array([self.label] * sequence_size)

            return data, time, label, self.example_id, self.context

    def _initialize(self, **kwargs):
        assert 0 < self.sequence_size <= 784, 'Bad sequence length %d' % self.sequence_size

    def _create_examples(self):
        x, y = self.load_mnist(self.data_path, self.data_type)

        # Unique labels are from 0 to 10
        labels = range(10)

        if self.data_type == self.Validation_Data or self.data_type == self.Train_Data:
            # Split out the data according to the CV fold

            start = int(self.cv_k / self.cv_n * len(x))
            end = int((self.cv_k + 1) / self.cv_n * len(x))
            logger.debug("Using CV split cv_n: %s, cv_k: %s, start: %s, end: %s" % (self.cv_n, self.cv_k, start, end))

            if self.data_type == self.Train_Data:
                x = np.concatenate((x[:start], x[end:]))
                y = np.concatenate((y[:start], y[end:]))
            else:
                x = x[start:end]
                y = y[start:end]

        # Create examples
        for i, label in enumerate(labels):
            x_one_class = x[y == label]

            self.examples.append([MnistDataReader.ExampleInfo(example_id=str((label, j)), data=x_sample, label=label)
                                  for (j, x_sample) in enumerate(x_one_class)])

            logger.debug('Label %s: Number of recordings %d, Cumulative Length %d' %
                         (label, len(self.examples[i]), sum([e.get_length() for e in self.examples[i]])))

        logger.debug('Number of sequences in the dataset %d' % nested_list_len(self.examples))

        # Additional data structure (faster access for some operations)
        for class_examples in self.examples:
            for example in class_examples:
                self.examples_dict[example.example_id] = example

    @staticmethod
    # Has to be a static method, context_size is required when creating the model,
    # DataReader can't be instantiated properly before the model is created
    def context_size(**kwargs):
        return 0

    @staticmethod
    def input_size(**kwargs):
        return 1

    @staticmethod
    def output_size(**kwargs):
        return 10

    @staticmethod
    def load_mnist(data_dir, data_type=SequenceDataReader.Train_Data):
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)

        data_file = os.path.join(data_dir, 'mnist.pkl.gz')
        if not os.path.exists(data_file):
            logger.info('Downloading MNIST from the web ...')
            try:
                import urllib
                urllib.urlretrieve('http://google.com')
            except AttributeError:
                import urllib.request as urllib
            url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
            urllib.urlretrieve(url, data_file)

        logger.info('Loading data ...')
        # Load the dataset
        f = gzip.open(data_file, 'rb')
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding="latin1")
        except TypeError:
            train_set, valid_set, test_set = pickle.load(f)
        f.close()

        if data_type == SequenceDataReader.Train_Data or data_type == SequenceDataReader.Validation_Data:
            # We might want to make a different validation split!
            x_1, y_1 = train_set
            x_2, y_2 = valid_set
            x = np.concatenate((x_1, x_2))
            y = np.concatenate((y_1, y_2))

        elif data_type == SequenceDataReader.Test_Data:
            x, y = test_set
        else:
            raise NotImplementedError('data_type %s is not implemented' % data_type)

        x = x.reshape(x.shape[0], 784, 1).astype(np.float32)
        y = y

        return x, y