import logging

import numpy as np

from src.data_reading.data_reader import SequenceDataReader
from src.utils import nested_list_len

logger = logging.getLogger(__name__)


class OneLabelDataReader(SequenceDataReader):
    NUM_OF_EXAMPLES = 32

    class ExampleInfo(SequenceDataReader.SequenceExampleInfo):
        MIN_DURATION = 60000
        MAX_DURATION = 80000

        def __init__(self, example_id, label):
            super().__init__(example_id)

            self.length = np.random.random_integers(self.MIN_DURATION, self.MAX_DURATION)
            self.data = np.random.normal(0, 1.0, size=[self.length, 22]).astype(np.float32)
            self.label = label

        def get_length(self):
            return self.length

        def read_data(self, serialized):
            index, sequence_size = serialized
            data = np.reshape(self.data[index: index + sequence_size], newshape=[sequence_size, 22])

            # Time might be needed for some advanced models like PhasedLSTM
            time = np.reshape(np.arange(index, index + sequence_size), newshape=[sequence_size, 1])

            label = np.array([self.label] * sequence_size)

            return data, time, label, self.example_id

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _create_examples(self):

        for label in [0, 1]:
            class_examples = []
            for i in range(self.NUM_OF_EXAMPLES):
                class_examples.append(OneLabelDataReader.ExampleInfo((label, i), label))
            self.examples.append(class_examples)

        logger.info('In class %d, cumulative length %d' % (0, sum([e.get_length() for e in self.examples[0]])))
        logger.info('Number of sequences in the dataset %d' % nested_list_len(self.examples))

        # Additional data structure for faster access
        # Move to the base
        for class_examples in self.examples:
            for example in class_examples:
                self.examples_dict[example.example_id] = example
