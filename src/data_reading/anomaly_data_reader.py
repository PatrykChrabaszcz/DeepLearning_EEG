from src.data_reading.data_reader import SequenceDataReader
import mne
import os
import numpy as np
from src.utils import nested_list_len, load_dict
import logging


logger = logging.getLogger(__name__)


# We assume that the data is available in a given structure:
# If you want to know how the data is prepared then
# look at the src/data_preparation/anomaly_data_generator.py
# data_path
#   train
#       data
#           00000003_s02_a01_Age_70_Gender_F_raw.fif
#       info
#           00000003_s02_a01_Age_70_Gender_F.p
#           ...
#   validation
#        ...
#   test
#       ...
#
# Files inside the 'data' directory contain the data (channel normalized). One file per one recording.
# Files inside the 'info' directory contain the metadata in a json format:
#   "age": 70,  - Age of the subject
#   "anomaly": 0, - Anomaly label for the recording (1 means that there was anomaly detected inside the data)
#   "file_name": "...", - File path to the original data file used to create corresponding .fif file
#   "gender": "F", - Gender of the subject
#   "number": "00000003", - Subject specific number
#   "sequence_name": "00000003_s02_a01" - Unique id that can be used to identify the recording
# Cast all prediction problems (all label_types) as  classification problem

class AnomalyDataReader(SequenceDataReader):
    label_types = ['age_class', 'anomaly', 'gender_class']

    class ExampleInfo(SequenceDataReader.SequenceExampleInfo):
        def __init__(self, info_dict, label_type, offset_size=0, limit_duration=None):
            super().__init__(example_id=info_dict['sequence_name'], offset_size=offset_size)

            # TODO save normalization in the dictionary
            self.age = (info_dict['age'] - 49.295620438) / 17.3674915241
            self.label = info_dict[label_type]
            self.mean = np.array(info_dict['mean'], dtype=np.float32)
            self.std = np.array(info_dict['std'], dtype=np.float32)

            self.file_handler = mne.io.read_raw_fif(info_dict['data_file'], preload=False, verbose='error')
            self.length = self.file_handler.n_times
            self.length = self.length if limit_duration is None else min(self.length, limit_duration)

            if label_type == 'age_class':
                self.context = np.array([info_dict['gender_class']]).astype(np.float32)
            elif label_type == 'anomaly':
                self.context = np.array([[info_dict['gender_class'], self.age]]).astype(np.float32)
            elif label_type == 'gender_class':
                self.context = np.array([self.age]).astype(np.float32)
            else:
                raise NotImplementedError('Can not create context for this label_type %s' % label_type)

        def get_length(self):
            return self.length

        def read_data(self, serialized):
            index, sequence_size = serialized
            data = self.file_handler.get_data(None, index, index + sequence_size).astype(np.float32)
            data = np.transpose(data)
            data = (data - self.mean) / self.std
            # Time might be needed for some advanced models like PhasedLSTM
            time = np.reshape(np.arange(index, index + sequence_size), newshape=[sequence_size, 1])

            label = np.array([self.label] * sequence_size)
            return data, time, label, self.example_id, self.context

    @staticmethod
    def add_arguments(parser):
        SequenceDataReader.add_arguments(parser)
        parser.add_argument("--label_type", type=str, dest='label_type', choices=AnomalyDataReader.label_types,
                            help="Path to the directory containing the data")

    def _initialize(self, label_type, limit_examples, limit_duration, **kwargs):
        self.label_type = label_type

        self.limit_examples = limit_examples
        self.limit_examples = None if self.limit_examples <= 0 else self.limit_examples

        self.limit_duration = limit_duration
        self.limit_duration = None if self.limit_duration <= 0 else self.limit_duration

        logger.debug('Initialized AnomalyDataReader with parameters:')
        logger.debug('label_type: %s' % self.label_type)
        logger.debug('limit_examples: %s' % self.limit_examples)
        logger.debug('limit_duration: %s' % self.limit_duration)

    def _create_examples(self):
        logger.info('Will create examples for this dataset.\n'
                    'Only the header info for each file is preloaded to the memory.\n'
                    'Parallel threads will read the data online.\n')

        # Train and Validation are located inside the 'train' folder
        if self.data_type == self.Validation_Data or self.data_type == self.Train_Data:
            folder_name = 'train'
        elif self.data_type == self.Test_Data:
            folder_name = 'test'
        else:
            raise NotImplementedError('data_type is not from the set {train, validation, test}')

        # Load data into dictionaries from the info json files
        info_dir = os.path.join(self.data_path, folder_name, 'info')
        info_files = sorted(os.listdir(info_dir))
        info_dicts = [load_dict(os.path.join(info_dir, i_f)) for i_f in info_files]

        for info_dict, info_file in zip(info_dicts, info_files):
            info_dict['data_file'] = os.path.join(self.data_path, folder_name, 'data', info_file[:-2] + '_raw.fif')
            # Compute additional fields (used for new labels and context information)
            info_dict['age_class'] = 1 if info_dict['age'] >= 49 else 0
            info_dict['gender_class'] = 1 if info_dict['gender'] == 'M' else 0

        # Find out what are the names of unique labels
        labels = list(set([info_dict[self.label_type] for info_dict in info_dicts]))

        logger.info('Will use %s as a label type' % self.label_type)

        logger.info('Create info objects for the files (Number of all sequences: %s' % len(info_dicts))

        if self.data_type == self.Validation_Data or self.data_type == self.Train_Data:
            # Split out the data according to the CV fold

            start = int(self.cv_k/self.cv_n * len(info_dicts))
            end = int((self.cv_k+1)/self.cv_n * len(info_dicts))
            logger.info("Using CV split cv_n: %s, cv_k: %s, start: %s, end: %s" % (self.cv_n, self.cv_k, start, end))

            validation_info_dicts = info_dicts[start:end]
            train_info_dicts = info_dicts[:start] + info_dicts[end:]

            if self.data_type == self.Train_Data:
                info_dicts = train_info_dicts
            else:
                info_dicts = validation_info_dicts

        for i, label in enumerate(labels):
            label_info_dicts = [info_dict for info_dict in info_dicts if info_dict[self.label_type] == label]
            label_info_dicts = label_info_dicts[:self.limit_examples]

            self.examples.append([AnomalyDataReader.ExampleInfo(label_info_dict, self.label_type,
                                                                self.offset_size, self.limit_duration)
                                  for (j, label_info_dict) in enumerate(label_info_dicts)])

            logger.debug('Label %s: Number of recordings %d, Cumulative Length %d' %
                         (label, len(self.examples[i]), sum([e.get_length() for e in self.examples[i]])))

        logger.info('Number of sequences in the dataset %d' % nested_list_len(self.examples))

        # Additional data structure (faster access for some operations)
        for class_examples in self.examples:
            for example in class_examples:
                self.examples_dict[example.example_id] = example

    @staticmethod
    # Has to be a static method, context_size is required when creating the model,
    # DataReader can't be instantiated properly before the model is created
    def context_size(label_type, **kwargs):
        if label_type in ['age_class', 'gender_class']:
            return 1
        elif label_type == 'anomaly':
            return 2

    @staticmethod
    def input_size(**kwargs):
        return 22

    @staticmethod
    def output_size(**kwargs):
        return 2
