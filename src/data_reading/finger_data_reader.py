from src.data_reading.data_reader import SequenceDataReader
import logging
import numpy as np
from scipy.io import loadmat
import os
import multiprocessing
from multiprocessing import sharedctypes
import ctypes
import random

logger = logging.getLogger(__name__)


class FingerDataReader(SequenceDataReader):
    class ExampleInfo(SequenceDataReader.SequenceExampleInfo):
        def __init__(self, example_id, data_ctypes, data_shape, labels_ctypes, labels_shape,
                     offset_size=0, limit_duration=None):
            super().__init__(example_id=example_id, offset_size=offset_size)

            self.data = np.frombuffer(data_ctypes, dtype=np.float32, count=int(np.prod(data_shape)))
            self.data.shape = data_shape
            self.labels = np.frombuffer(labels_ctypes, dtype=np.float32, count=int(np.prod(labels_shape)))
            self.labels.shape = labels_shape

            self.length = self.data.shape[0]
            self.length = self.length if limit_duration is None else min(self.length, limit_duration)

            self.context = None

        def get_length(self):
            return self.length

        def read_data(self, serialized):
            index, sequence_size = serialized

            data = self.data[index: index+sequence_size]
            # Time might be needed for some advanced models like PhasedLSTM
            time = np.reshape(np.arange(index, index + sequence_size), newshape=[sequence_size, 1])

            label = self.labels[index: index + sequence_size]

            return data, time, label, self.example_id, self.context

        # We need to overwrite this method, all examples read from the same array, we need to read from a
        # completely different points within the sequence
        def reset(self, sequence_size, randomize=False, new_epoch=False):
            super().reset(sequence_size, randomize, new_epoch)

            # Additional behaviour (overwrite the self.curr_index with random point in the file (not just a phase)
            if randomize and not self.done:
                margin = self.get_length() - self.sequence_size
                self.curr_index = random.randint(0, margin)

    @staticmethod
    def add_arguments(parser):
        SequenceDataReader.add_arguments(parser)
        parser.add_argument("--subject", type=int, dest='subject', choices=[1, 2, 3], default=1,
                            help="Number of the patient.")
        parser.add_argument("--split", type=float, dest='split', default=0.8,
                            help="Train/Validation split, if 0.8 then 80% of data is used for training.")

    def _initialize(self, subject, split, **kwargs):
        self.subject = subject
        self.split = split

        if self.balanced:
            logger.warn('Class balancing is not implemented for this dataset.')

    def _create_examples(self):
        logger.info('Read information about all sample files from the dataset')

        files = [os.path.join(self.data_path, 'sub%d_comp.mat' % subject) for subject in range(1, 4)]

        file = files[self.subject]
        data_dict = loadmat(file)
        data = data_dict['train_data'].astype(np.float32)
        labels = data_dict['train_dg'].astype(np.float32)
        split_point = int(data.shape[0] * self.split)

        # First x% of data is used for training, last (100-x)% of data is used for validation
        if self.data_type == SequenceDataReader.Train_Data:
            data = np.ascontiguousarray(data[:split_point], dtype=np.float32)
            labels = np.ascontiguousarray(labels[:split_point], dtype=np.float32)
        elif self.data_type == SequenceDataReader.Validation_Data:
            data = np.ascontiguousarray(data[split_point:], dtype=np.float32)
            labels = np.ascontiguousarray(labels[split_point:], dtype=np.float32)
        else:
            raise RuntimeError('Test split is not implemented right now for this dataset (Labels are not provided)')

        # Read only access so no need to lock
        data_shape = data.shape
        labels_shape = labels.shape
        data.shape = data.size
        labels.shape = labels.size

        self.data_ctypes = sharedctypes.RawArray('f', data)
        self.labels_ctypes = sharedctypes.RawArray('f', labels)

        logger.info('Create info objects for the files')

        # For validation and test we only have one example
        number_of_examples = 128 if self.data_type == SequenceDataReader.Train_Data else 1
        self.examples.append([FingerDataReader.ExampleInfo(i, self.data_ctypes, data_shape, self.labels_ctypes,
                                                           labels_shape, offset_size=self.offset_size)
                              for i in range(number_of_examples)])

        # Additional data structure for faster access
        for class_examples in self.examples:
            for example in class_examples:
                self.examples_dict[example.example_id] = example

    @staticmethod
    def context_size(**kwargs):
        return 0

    @staticmethod
    def input_size(**kwargs):
        return 48

    @staticmethod
    def output_size(**kwargs):
        return 5
