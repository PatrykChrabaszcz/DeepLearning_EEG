from src.data_reading.data_reader import SequenceDataReader
import mne
import os
import numpy as np
from src.utils import nested_list_len, load_dict
import logging
import random


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
    label_age = 'age_class'
    label_anomaly = 'anomaly'
    label_gender = 'gender_class'
    label_types = [label_age, label_anomaly, label_gender]

    normalization_none = 'none'
    normalization_separate = 'separate'
    normalization_types = [normalization_none, normalization_separate]

    class ExampleInfo(SequenceDataReader.SequenceExampleInfo):
        def __init__(self, info_dict, label_type, offset_size=0, random_mode=0, limit_duration=None,
                     use_augmentation=0):
            super().__init__(example_id=info_dict['sequence_name'], offset_size=offset_size, random_mode=random_mode)

            self.use_augmentation = use_augmentation
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

        def reset(self, sequence_size):
            super().reset(sequence_size)

        # Scale from 0.5 to 2.0
        @staticmethod
        def random_scale():
            r = random.uniform(-1, 1)
            return 2**r

        def get_length(self):
            return self.length

        def read_data(self, serialized):
            index, sequence_size = serialized
            data = self.file_handler.get_data(None, index, index+ sequence_size).astype(np.float32)
            data = np.transpose(data)

            # Now data has shape time x features

            # Normalize
            data = ((data - self.mean) / self.std)

            # Rescale by random value from log_uniform (0.5, 2) and revert in time with 50% chance
            if self.use_augmentation:
                data *= self.random_scale()

                revert = random.choice([0, 1])
                if revert:
                    data = data[::-1, :]

            # Time might be needed for some advanced models like PhasedLSTM
            time = np.reshape(np.arange(index, index + sequence_size), newshape=[sequence_size, 1])

            label = np.array([self.label] * sequence_size)
            return data, time, label, self.example_id, self.context

    @staticmethod
    def add_arguments(parser):
        SequenceDataReader.add_arguments(parser)
        parser.add_argument("label_type", type=str, dest='label_type', choices=AnomalyDataReader.label_types,
                            help="Path to the directory containing the data")
        parser.add_argument("normalization_type", type=str, dest='normalization_type',
                            choices=AnomalyDataReader.normalization_types,
                            help="How to normalize the data.")
        parser.add_argument("use_augmentation", type=int, choices=[0, 1], default=0,
                            help="Will add some shift and rescale the data."
                                 "Idea is to make the network invariant to those transformations")
        parser.add_argument("filter_gender", type=str, choices=['None', 'M', 'F'], default='None',
                            help="Will train using only male of female ")
        return parser

    def _initialize(self, label_type, normalization_type, use_augmentation, filter_gender, **kwargs):
        self.label_type = label_type
        self.normalization_type = normalization_type
        self.use_augmentation = use_augmentation
        self.filter_gender = filter_gender

        if use_augmentation and self.data_type != SequenceDataReader.Train_Data:
            logger.warning('For Validation and Test we disable data augmentation')
            self.use_augmentation = 0

        self.limit_examples = None if self.limit_examples <= 0 else self.limit_examples
        self.limit_duration = None if self.limit_duration <= 0 else self.limit_duration

        logger.debug('Initialized AnomalyDataReader with parameters:')
        logger.debug('label_type: %s' % self.label_type)
        logger.debug('limit_examples: %s' % self.limit_examples)
        logger.debug('limit_duration: %s' % self.limit_duration)

    @staticmethod
    def load_info_dicts(data_path, folder_name):
        info_dir = os.path.join(data_path, folder_name, 'info')
        info_files = sorted(os.listdir(info_dir))
        info_dicts = [load_dict(os.path.join(info_dir, i_f)) for i_f in info_files]

        for info_dict, info_file in zip(info_dicts, info_files):
            info_dict['data_file'] = os.path.join(data_path, folder_name, 'data', info_file[:-2] + '_raw.fif')
            # Compute additional fields (used for new labels and context information)
            info_dict['age_class'] = 1 if info_dict['age'] >= 49 else 0
            info_dict['gender_class'] = 1 if info_dict['gender'] == 'M' else 0

        return info_dicts

    def _create_examples(self):
        if self.normalization_type != self.normalization_none:
            if self.limit_duration != None:
                logger.warning('Will limit example duration but will compute normalization statistics from full '
                               'recordings.')
        if self.normalization_type != self.normalization_separate:
            if self.use_augmentation:
                logger.warning('Augmentation was only designed for \"separate\" normalization.')

        # Train and Validation are located inside the 'train' folder
        if self.data_type == self.Validation_Data or self.data_type == self.Train_Data:
            folder_name = 'train'
        elif self.data_type == self.Test_Data:
            folder_name = 'test'
        else:
            raise NotImplementedError('data_type is not from the set {train, validation, test}')

        # Load data into dictionaries from the info json files
        info_dicts = self.load_info_dicts(self.data_path, folder_name)

        # Filter out based on the gender if that is what user wants
        if self.filter_gender in ['M', 'F']:
            logger.warning('Will only use recordings with gender: %s' % self.filter_gender)
            info_dicts = [info_dict for info_dict in info_dicts if info_dict['gender'] == self.filter_gender]

        # Find out what are the names of unique labels
        labels = list(set([info_dict[self.label_type] for info_dict in info_dicts]))

        if len(labels) != self.output_size():
            error = 'Not all labels present in the dataset, declared %d, detected %d' % (self.output_size(), len(labels))
            logger.error(error)
            raise RuntimeError(error)

        logger.debug('Will use %s as a label type' % self.label_type)

        logger.debug('Create info objects for the files (Number of all sequences: %s' % len(info_dicts))

        if self.data_type == self.Validation_Data or self.data_type == self.Train_Data:
            # Split out the data according to the CV fold

            start = int(self.cv_k/self.cv_n * len(info_dicts))
            end = int((self.cv_k+1)/self.cv_n * len(info_dicts))
            logger.debug("Using CV split cv_n: %s, cv_k: %s, start: %s, end: %s" % (self.cv_n, self.cv_k, start, end))

            validation_info_dicts = info_dicts[start:end]
            train_info_dicts = info_dicts[:start] + info_dicts[end:]

            if self.data_type == self.Train_Data:
                info_dicts = train_info_dicts
            else:
                info_dicts = validation_info_dicts

        if self.normalization_type == self.normalization_none:
            logger.debug('Will not normalize the data.')
            for info_dict in info_dicts:
                info_dict['mean'] = 0.0
                info_dict['std'] = 1.0
        elif self.normalization_type == self.normalization_separate:
            logger.debug('Will normalize each recording separately.')
        else:
            raise NotImplementedError('This normalization (%s) is not implemented' % self.normalization_type)

        # Create examples
        for i, label in enumerate(labels):
            label_info_dicts = [info_dict for info_dict in info_dicts if info_dict[self.label_type] == label]
            label_info_dicts = label_info_dicts[:self.limit_examples]

            self.examples.append([AnomalyDataReader.ExampleInfo(label_info_dict, self.label_type,
                                                                self.offset_size, self.random_mode,
                                                                self.limit_duration, self.use_augmentation)
                                  for (j, label_info_dict) in enumerate(label_info_dicts)])

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
    def context_size(label_type, **kwargs):
        if label_type in ['age_class', 'gender_class']:
            return 1
        elif label_type == 'anomaly':
            return 2

    @staticmethod
    def input_size(**kwargs):
        return 21

    @staticmethod
    def output_size(**kwargs):
        return 2

