import mne
import os
import numpy as np
import multiprocessing
from src.utils import Stats, nested_list_len
import random
import logging
import signal
from queue import Empty

# Definitions
# EXAMPLE: we use the term 'example' to define independent sequence, e.g. full sequence recording from one session.
# SEQUENCE: example can be divided into separate (dependent) sequences, e.g. example with 1000 samples can be divided
# into 10 sequences with size 100.
#
# We batch sequences from different (random) examples and use them for training.
# If we have an example of size 300(t: 0-299) with sequences of size 100 : A (t: 0-99), B(t: 100-199), C(199: 299)
# then we take care of forwarding RNN state from the end of sequence A to the beginning if sequence B.
# Because of that it is impossible to put 2 sequences from the same example into one mini-batch.


logger = logging.getLogger(__name__)


# We need to manage state
class SequenceDataReader:
    class EpochDone(Exception):
        pass

    Train_Data = 'train'
    Validation_Data = 'validation'
    Test_Data = 'test'

    # This class is used to hold information about examples
    class SequenceExampleInfo:
        def __init__(self, example_id):
            # example_id is used to identify current RNN state for specific example.
            # We want to have an ability to process batches with random samples, but we still need to
            # use proper initial RNN state in each mini-batch iteration
            self.example_id = example_id
            self.sequence_size = None

            # Has to be initialized at the beginning of every epoch
            self.state = None

            self.curr_index = 0
            self.done = False
            self.blocked = False

        # Resets example to the start position, can also change sequence size on reset
        def reset(self, sequence_size, randomize=False):
            self.sequence_size = sequence_size

            self.blocked = False

            # If sample is smaller than sequence size then we simply will ignore it
            margin = self.get_length() - self.sequence_size
            if margin < 0:
                self.done = True
                return

            self.done = False
            # Randomly shifts the sequence (Phase)
            self.curr_index = 0
            if randomize:
                self.curr_index = random.randint(0, min(margin, self.sequence_size))

        # What is uploaded to the info_queue
        def get_info_and_advance(self):
            if self.sequence_size is None:
                raise RuntimeError('Can\'t use example if reset() was not called!')
            if self.done or self.blocked:
                raise RuntimeError('Impossible to get the data for already done or blocked example')

            index = self.curr_index
            self.curr_index += self.sequence_size

            # If we can't extract more samples then we are done
            self.done = self.curr_index + self.sequence_size > self.get_length()
            self.blocked = True

            return self.example_id, (index, )

        def get_length(self):
            raise NotImplementedError

        def read_data(self, serialized):
            raise NotImplementedError

    def __init__(self, readers_count, state_initializer, data_type=Train_Data, batch_size=64,
                 balanced=True, allow_smaller_batch=False):

        # state_initializer function is used to get the initial state (for example: random or zero)
        self.state_initializer = state_initializer
        # If balanced then will take the same amount of examples from each class
        self.balanced = balanced

        # Train or Validation type
        self.data_type = data_type

        self.batch_size = batch_size

        self.allow_smaller_batch = allow_smaller_batch

        # Info Queue -> Information that is used to read a proper chunk of data from a proper file
        self.info_queue = multiprocessing.Queue()

        # Data Queue -> Data with labels
        self.data_queue = multiprocessing.Queue()

        self.readers_count = readers_count
        self.readers = []
        self.stop_readers_event = multiprocessing.Event()

        # Number of samples to read in the current epoch
        self.samples_count = 0

        # self.examples[i] -> List with examples for class i
        self.examples = []
        # dictionary example_id -> example for faster access when updating/receiving RNN state
        self.examples_dict = {}

        # Will populate examples list and dictionary with all examples
        self._create_examples()
        self._create_readers()

        self.state_needs_update = True

    # Implements a method that will create file reader threads, and append them to the self.readers list
    def _create_readers(self):
        logger.info('Create reader processes')
        for i in range(self.readers_count):
            p = multiprocessing.Process(target=SequenceDataReader.read_sample_function,
                                        args=(self.info_queue, self.data_queue, self.examples_dict))
            self.readers.append(p)

    def stop_readers(self):
        logger.info('Trying to stop the readers ...')

        while self.info_queue.qsize() > 0:
            try:
                self.info_queue.get(timeout=1)
            except Empty:
                logger.debug('During cleanup, trying to get an element when queue is empty')

        # Will stop readers if they are blocked on the input read
        for i in range(self.readers_count):
            self.info_queue.put((None, None))

        # This is super strange to me:
        # If output_queue has more than 1 element then somehow we are not able to join those processes

        # Solution: Simply clear out that data queue to make it possible to nicely shut down without zombie processes
        while self.data_queue.qsize() > 0:
            self.data_queue.get(timeout=1)

        for (i, r) in enumerate(self.readers):
            logger.info('Waiting on join (%s) for reader %d' % (self.data_type, i))
            r.join()

        logger.info('Readers joined')

    # Will stop when stop_readers_event is set unless info_queue is empty
    # If info_queue is empty then will stop if receives None as example_id
    @staticmethod
    def read_sample_function(info_queue, output_queue, examples_dict):
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        logger.info('New reader process is running ...')

        while True:
            example_id, serialized = info_queue.get()
            if example_id is not None:
                data = examples_dict[example_id].read_data(serialized)
                output_queue.put(data)

            else:
                logger.info('Reader received None, finishing the process ...')
                break

        logger.info('Reader process finished.')

    # Implements a method that will fill up self.examples list. This list contains all necessary information about
    # all examples
    def _create_examples(self):
        raise NotImplementedError

    def _get_random_example(self, class_label):
        # Find all not done examples for this class
        all_class_examples = [e for e in self.examples[class_label] if (not e.done) and (not e.blocked)]
        if len(all_class_examples) == 0:
            return None
        v = random.choice(all_class_examples)
        return v

    # This should append up to 'size' samples to the info_queue. We make sure that at a given point in time at most
    # one sequence from each example is inside the info queue and data_queue
    def _append_samples(self, size):
        if self.balanced:
            assert size % len(self.examples) == 0
            s = size // len(self.examples)

            for i in range(s):
                examples = [self._get_random_example(class_label) for class_label in range(len(self.examples))]
                if None in examples:
                    return

                for example in examples:
                    self.samples_count += 1
                    self.info_queue.put(example.get_info_and_advance())

        else:
            for i in range(size):
                class_labels = [i for (i, e) in enumerate(self.examples) if len(e)]
                if len(class_labels) is 0:
                    return

                example = random.choice(class_labels)
                self.samples_count += 1
                self.info_queue.put(example.get_info_and_advance())

    def initialize_epoch(self, sequence_size, randomize):
        logger.debug('Initialize new epoch (%s)' % self.data_type)
        # Read examples that we were unable to process
        for i in range(self.samples_count):
            self.data_queue.get()
            self.samples_count -= 1

        assert self.samples_count == 0

        for class_examples in self.examples:
            for example in class_examples:
                example.reset(sequence_size, randomize)
                example.state = self.state_initializer()

        # Populate info queue with some examples
        self._append_samples(5 * self.batch_size)

        self.state_needs_update = False

    def set_states(self, keys, states):
        assert(len(keys) == len(states))

        for key, state in zip(keys, states):
            self.examples_dict[key].state = state
            self.examples_dict[key].blocked = False

        self._append_samples(self.batch_size)
        self.state_needs_update = False

    # Prepare list of states for given keys, if forget is True then use initial states instead
    # (No forward state propagation during truncated back-propagation)
    def get_states(self, keys, forget=False):
        states = []
        for key in keys:
            if forget:
                states.append(self.state_initializer())
            else:
                states.append(self.examples_dict[key].state)
        return states

    def start_readers(self):
        logger.info('Starting readers')
        for r in self.readers:
            r.start()

    def get_batch(self):
        if self.state_needs_update:
            raise RuntimeError('State needs an update')

        data_arrays = []
        time_arrays = []
        labels = []
        ids = []

        if self.samples_count < self.batch_size:
            if not self.allow_smaller_batch or self.samples_count == 0:
                raise SequenceDataReader.EpochDone
            else:
                batch_size = self.samples_count
        else:
            batch_size = self.batch_size

        for i in range(batch_size):
            data, time, label, example_id = self.data_queue.get()
            self.samples_count -= 1
            data_arrays.append(data)
            time_arrays.append(time)
            labels.append(label)
            ids.append(example_id)

        self.state_needs_update = True
        return np.stack(data_arrays, axis=0), np.stack(time_arrays, axis=0), np.array(labels), ids


class AnomalyDataReader(SequenceDataReader):
    class ExampleInfo(SequenceDataReader.SequenceExampleInfo):
        def __init__(self, example_id, file_name, label, limit_duration=None):
            super().__init__(example_id)

            self.file_name = file_name
            self.label = label

            self.file_handler = mne.io.read_raw_fif(file_name, preload=False, verbose='error')
            self.length = self.file_handler.n_times
            self.length = self.length if limit_duration is None else min(self.length, limit_duration)

        def get_length(self):
            return self.length

        def read_data(self, serialized):
            index = serialized[0]
            data = self.file_handler.get_data(None, index, index + self.sequence_size).astype(np.float32)
            data = np.transpose(data)
            # Time might be needed for some advanced models like PhasedLSTM
            time = np.reshape(np.arange(index, index + self.sequence_size), newshape=[self.sequence_size, 1])

            label = np.array([self.label] * self.sequence_size)
            return data, time, label, self.example_id

    def __init__(self, cache_path, limit_examples=None, limit_duration=None, **kwargs):
        self.cache_path = cache_path
        self.input_dim = 22
        self.limit_examples = limit_examples
        self.limit_duration = limit_duration
        super().__init__(**kwargs)

    def _create_examples(self):
        logger.info('Read information about all sample files from the dataset')
        p = os.path.join(self.cache_path, self.data_type, 'normal')
        file_normal_paths = [os.path.join(p, f) for f in os.listdir(p)]
        p = os.path.join(self.cache_path, self.data_type, 'abnormal')
        file_abnormal_paths = [os.path.join(p, f) for f in os.listdir(p)]

        if self.limit_examples is not None:
            file_normal_paths = file_normal_paths[:self.limit_examples]
            file_abnormal_paths = file_abnormal_paths[:self.limit_examples]

        logger.info('Report for %s:' % self.data_type)
        logger.info('Number of normal recordings: %d' % len(file_normal_paths))
        logger.info('Number of abnormal recordings: %d' % len(file_abnormal_paths))

        logger.info('Create info objects for the files')

        for i, file_paths in enumerate([file_normal_paths, file_abnormal_paths]):
            self.examples.append([AnomalyDataReader.ExampleInfo((i, j), fn, i, self.limit_duration)
                                  for (j, fn) in enumerate(file_paths)])
            logger.info('In class %d, cumulative length %d' % (i, sum([e.get_length() for e in self.examples[-1]])))

        logger.info('Number of sequences in the dataset %d' % nested_list_len(self.examples))

        # Additional data structure for faster access
        for class_examples in self.examples:
            for example in class_examples:
                self.examples_dict[example.example_id] = example


if __name__ == '__main__':

    pass