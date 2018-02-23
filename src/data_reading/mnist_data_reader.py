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



# from src.data_reading.data_reader import SequenceDataReader
# import mne
# import os
# import numpy as np
# from src.utils import nested_list_len, load_dict
# import logging
# from torchvision import datasets
# import gzip
# import pickle
#
# logger = logging.getLogger(__name__)
#
#
# # This does not use multi-threading at all
# class MnistDataReader(SequenceDataReader):
#     @staticmethod
#     def add_arguments(parser):
#         # No arguments from the parent
#         # SequenceDataReader.add_arguments(parser)
#         parser.add_argument("--data_path", type=str, dest='data_path',
#                             help="Path to the directory containing the data")
#         parser.add_argument("--sequence_size", type=int, dest='sequence_size', default=784,
#                             help="Sequence size")
#         parser.add_argument("--batch_size", type=int, dest='batch_size', default=64,
#                             help="Batch size used for training")
#         parser.add_argument("--forget_state", type=int, dest='forget_state', default=1, choices=[0, 1],
#                             help="Batch size used for training")
#         parser.add_argument("--random_mode", type=int, dest='random_mode', default=0, choices=[0, 1, 2],
#                             help="0 - No randomization; 1 - Reads sequentially but each time starts recording from a "
#                                  "new offset; 2 - Reads random chunks, should not be used if forget_state=False."
#                                  "Applies only to the train data reader, validation data reader has it set to 0.")
#         return parser
#
#     def __init__(self, data_path, batch_size, sequence_size, random_mode, state_initializer, allow_smaller_batch=False,
#                  data_type=SequenceDataReader.Train_Data, **kwargs):
#         # Does not call parent constructor
#         assert sequence_size == 784, 'For Mnist only 784 sequence size is available'
#
#         self.batch_size = batch_size
#         self.random_mode = random_mode
#         self.state_initializer = state_initializer
#         self.allow_smaller_batch = allow_smaller_batch
#         self.data_type = data_type
#
#         self.index = 0
#
#         self.data, self.labels = self.load_mnist(data_path, data_type)
#
#         logger.info('Loaded the data')
#
#     def initialize_epoch(self, sequence_size):
#         assert sequence_size == 784, 'Only sequence size 784 is available for this dataset'
#         logger.info('Initialize new epoch (%s)' % self.data_type)
#
#     def set_states(self, keys, states):
#         # Nothing to do here
#         assert(len(keys) == len(states))
#
#     # Prepare list of states for given keys, if forget is True then use initial states instead
#     # (No forward state propagation during truncated back-propagation)
#     def get_states(self, keys, forget=False):
#         states = [self.state_initializer() for key in keys]
#         return states
#
#     def start_readers(self):
#         logger.info('Starting readers (This dataset does not use multi-threading).')
#
#     def stop_readers(self):
#         logger.info('Stop readers (This dataset does not use multi-threading).')
#
#     def get_batch(self):
#         samples_count = len(self.data) - self.index
#         if samples_count < self.batch_size:
#             if not self.allow_smaller_batch or samples_count == 0:
#                 self.index = 0
#                 # Optionally reshuffle
#                 raise SequenceDataReader.EpochDone
#             else:
#                 batch_size = samples_count
#         else:
#             batch_size = self.batch_size
#
#         data = self.data[self.index:self.index + batch_size, :].astype(np.float32)
#         labels = np.stack([np.array([l] * 784) for l in self.labels[self.index:self.index + batch_size]], axis=0)
#         time = np.stack([np.arange(0, 784) for _ in range(batch_size)]).astype(np.float32)
#
#         ids = [0 for _ in range(batch_size)]
#         context_arrays = [None for _ in range(batch_size)]
#
#         self.index += batch_size
#         return ids, data, time, labels, np.stack(context_arrays, axis=0)
#
#     @staticmethod
#     # Has to be a static method, context_size is required when creating the model,
#     # DataReader can't be instantiated properly before the model is created
#     def context_size(**kwargs):
#         return 0
#
#     @staticmethod
#     def input_size(**kwargs):
#         return 1
#
#     @staticmethod
#     def output_size(**kwargs):
#         return 10
#
#     @staticmethod
#     def labels(**kwargs):
#         return [i for i in range(10)]
#
#     def load_mnist(self, data_dir, data_type=SequenceDataReader.Train_Data):
#         if not os.path.exists(data_dir):
#             os.mkdir(data_dir)
#
#         data_file = os.path.join(data_dir, 'mnist.pkl.gz')
#         if not os.path.exists(data_file):
#             logger.info('Downloading MNIST from the web ...')
#             try:
#                 import urllib
#                 urllib.urlretrieve('http://google.com')
#             except AttributeError:
#                 import urllib.request as urllib
#             url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
#             urllib.urlretrieve(url, data_file)
#
#         logger.info('Loading data ...')
#         # Load the dataset
#         f = gzip.open(data_file, 'rb')
#         try:
#             train_set, valid_set, test_set = pickle.load(f, encoding="latin1")
#         except TypeError:
#             train_set, valid_set, test_set = pickle.load(f)
#         f.close()
#
#         if data_type == self.Train_Data:
#             x, y = train_set
#         elif data_type == self.Validation_Data:
#             x, y = valid_set
#         elif data_type == self.Test_Data:
#             x, y = test_set
#         else:
#             raise NotImplementedError('data_type %s is not Impelemented' % data_type)
#
#         x = x.reshape(x.shape[0], 784, 1).astype(np.float32)
#         y = y
#
#         return x, y
#
#
#
#     # def stochastic_data(self, x, y, batch_size):
#     #     idxs = np.random.permutation(x.shape[0])[:batch_size]
#     #     x = x[idxs]
#     #     y = y[idxs]
#     #
#     #     x = np.reshape(x, (x.shape[0], -1))
#     #     y = self.one_hot(y)
#     #     return x, y
#     #
#     #
#     # def full_data_generator(x, y, batch_size):
#     #     x = np.reshape(x, (x.shape[0], -1))
#     #     num = x.shape[0]
#     #     iter = int(num/batch_size)
#     #     for i in range(iter):
#     #         x_ = x[i*batch_size:(i+1)*batch_size]
#     #         y_ = y[i*batch_size:(i+1)*batch_size]
#     #         y_ = one_hot(y_)
#     #         yield x_, y_
#     #     if num % batch_size != 0:
#     #         x_ = x[iter*batch_size:num]
#     #         y_ = y[iter*batch_size:num]
#     #         y_ = one_hot(y_)
#     #         yield x_, y_
#     #
#     #
#     # def full_stochastic_generator(x, y, batch_size):
#     #     # Shuffle
#     #     idxs = np.random.permutation(x.shape[0])
#     #     x = x[idxs]
#     #     y = y[idxs]
#     #
#     #     return full_data_generator(x, y, batch_size)
#     #
#     #
#     # def one_hot(y):
#     #     size = 10
#     #     array = np.zeros(shape=(y.shape[0], size))
#     #
#     #     for c in range(size):
#     #         array[y == c, c] = 1
#     #
#     #     return array