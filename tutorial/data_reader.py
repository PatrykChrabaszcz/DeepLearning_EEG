import logging
import os

import numpy as np

from src.data_reading.data_reader import SequenceDataReader
from src.utils import nested_list_len

logger = logging.getLogger(__name__)


class TutorialDataReader(SequenceDataReader):
    NUM_OF_EXAMPLES = 64
    MIN_DURATION = 1000
    MAX_DURATION = 2000

    class TutorialExampleInfo(SequenceDataReader.SequenceExampleInfo):
        DELAY = 5

        def __init__(self, example_id, file_name):
            super().__init__(example_id)

            # Preload the data, for bigger files it might be a good idea to read chunks straight from the files inside
            # the read_data() function
            self.data = np.load(file_name).astype(np.float32)

            self.length = self.data.shape[0]

        def get_length(self):
            return self.length

        def read_data(self, serialized):
            index = serialized[0]
            data = np.reshape(self.data[index: index + self.sequence_size], newshape=[self.sequence_size, 1])

            # Time might be needed for some advanced models like PhasedLSTM
            time = np.reshape(np.arange(index, index + self.sequence_size), newshape=[self.sequence_size, 1])

            # Our network will have to output what was in the sequence 'DELAY' timestamps ago

            # For some fields at the beginning there are no labels, we will generate random ones
            extend = max(0, self.DELAY - index)
            if extend > 0:
                label = np.concatenate((
                    np.random.choice([-1.0, 1.0], size=min(extend, self.sequence_size)),
                    self.data[0:max(0, (self.sequence_size-extend))])).astype(dtype=np.float32)

            else:
                label = self.data[index-self.DELAY: index-self.DELAY+self.sequence_size]

            assert(len(label) == self.sequence_size)
            return data, time, label, self.example_id

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _create_examples(self):
        self.data_path = 'data/train' if self.data_type == SequenceDataReader.Train_Data else 'data/test'
        # If data is not present, generate some random sequences of -1 and 1
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
            logger.info('Generating the data')
            for i in range(self.NUM_OF_EXAMPLES):
                size = np.random.random_integers(self.MIN_DURATION, self.MAX_DURATION)
                data = np.random.choice([1.0, -1.0], size=size)
                data = np.array(data, np.float32)
                np.save(os.path.join(self.data_path, '%d.npy' % i), data)

        class_one_examples = []
        for i, file_name in enumerate(os.listdir(self.data_path)):
            file_name = os.path.join(self.data_path, file_name)
            class_one_examples.append(TutorialDataReader.TutorialExampleInfo(i, file_name))

        self.examples.append(class_one_examples)

        logger.info('In class %d, cumulative length %d' % (0, sum([e.get_length() for e in self.examples[0]])))
        logger.info('Number of sequences in the dataset %d' % nested_list_len(self.examples))

        # Additional data structure for faster access
        # Move to the base
        for class_examples in self.examples:
            for example in class_examples:
                self.examples_dict[example.example_id] = example
