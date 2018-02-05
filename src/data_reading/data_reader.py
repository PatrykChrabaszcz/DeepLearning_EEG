import numpy as np
import multiprocessing
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
        def __init__(self,
                     example_id,
                     offset_size):
            # example_id is used to identify current RNN state for specific example.
            # We want to have an ability to process batches with random samples, but we still need to
            # use proper initial RNN state in each mini-batch iteration
            self.example_id = example_id
            # offset_size is useful when CNNs are in the first couple of layers, if CNN decreases time resolution
            # then offset_size makes sure that hidden states are matched
            self.offset_size = offset_size
            self.sequence_size = None

            # Has to be initialized at the beginning of every epoch
            self.state = None

            self.curr_index = 0
            self.done = False
            self.blocked = False

        # Resets example to the start position, can also change sequence size on reset
        def reset(self, sequence_size, randomize=False, new_epoch=False):
            assert sequence_size is not None

            # Assert that we do not go back more than we go forward
            assert self.offset_size < sequence_size

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
            self.curr_index += self.sequence_size - self.offset_size

            # If we can't extract more samples then we are done

            self.done = self.curr_index + self.sequence_size > self.get_length()

            self.blocked = True

            return self.example_id, (index, self.sequence_size)

        def get_length(self):
            raise NotImplementedError

        def read_data(self, serialized):
            raise NotImplementedError

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--data_path", type=str, dest='data_path',
                            help="Path to the directory containing the data")
        parser.add_argument("--readers_count", type=int, dest='readers_count', default=3,
                            help="Learning rate used for training")
        parser.add_argument("--batch_size", type=int, dest='batch_size', default=64,
                            help="Learning rate used for training")
        parser.add_argument("--sequence_size", type=int, dest='sequence_size', default=1000,
                            help="How many time-points are used for one training sequence.")
        parser.add_argument("--balanced", type=int, dest='balanced', default=1, choices=[0, 1],
                            help="If greater than 0 then balance mini-batches.")
        parser.add_argument("--limit_examples", type=int, dest='limit_examples', default=0,
                            help="If greater than 0 then will only use this many examples per class.")
        parser.add_argument("--limit_duration", type=int, dest='limit_duration', default=0,
                            help="If greater than 0 then each example will only use first limit_duration samples.")
        parser.add_argument("--forget_state", type=int, dest='forget_state', default=0, choices=[0, 1],
                            help="If set to 1 then state will not be forward propagated between subsequences from the "
                                 "same example.")
        parser.add_argument("--cv_n", type=int, dest='cv_n', default=5,
                            help="How many folds are used for cross validation.")
        parser.add_argument("--cv_k", type=int, dest='cv_k', default=4,
                            help="Which fold is used for validation. Indexing starts from 0!")

    def __init__(self,
                 data_path,
                 state_initializer,
                 readers_count=1,
                 offset_size=0,
                 data_type=Train_Data,
                 batch_size=64,
                 balanced=True,
                 allow_smaller_batch=False,
                 continuous=False,
                 cv_n=5,
                 cv_k=4,
                 **kwargs):

        self.data_path = data_path
        # state_initializer function is used to get the initial state (for example: random or zero)
        self.state_initializer = state_initializer
        self.offset_size = offset_size
        # If balanced then will take the same amount of examples from each class
        self.balanced = balanced

        # Train or Validation type
        self.data_type = data_type

        self.batch_size = batch_size

        self.allow_smaller_batch = allow_smaller_batch

        # If set to true then after example is finished it will be immediately reset
        self.continuous = continuous

        self.cv_n = cv_n
        self.cv_k = cv_k
        assert cv_k < cv_n, "Fold used for validation has index which is higher than the number of folds."

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

        self.state_needs_update = True

        # If we are in continuous mode then we will ignore initialize_epoch() calls after the first one was done
        self.epoch_initialized = False
        self._last_sequence_size = None
        self._last_randomize = None

        self._initialize(**kwargs)

        self._create_examples()
        self._create_readers()

    # This should be used instead of constructor for the derived class
    # Is this a good design pattern??
    def _initialize(self, **kwargs):
        raise NotImplementedError('Implement instead of constructor')

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

    @staticmethod
    def input_size(**kwargs):
        return NotImplementedError('Trying to call input_size on the SequenceDataReader base class')

    @staticmethod
    def output_size(**kwargs):
        return NotImplementedError('Trying to call output_size on the SequenceDataReader base class')

    # Implements a method that will fill up self.examples list. This list contains all necessary information about
    # all examples
    def _create_examples(self):
        raise NotImplementedError

    # Probably this function could have lower complexity [Optimize if it takes too much time]
    def _get_random_example(self, class_label=None):
        if class_label is None:
            class_count = len(self.examples)
            all_examples = []
            for class_label in range(class_count):
                all_examples.extend([e for e in self.examples[class_label] if (not e.done) and (not e.blocked)])

        else:
            # Find all not done and not blocked examples for this class
            all_examples = [e for e in self.examples[class_label] if (not e.done) and (not e.blocked)]

        if len(all_examples) == 0:
            return None

        v = random.choice(all_examples)
        return v

    # This should append up to 'size' samples to the info_queue. We make sure that at a given point in time at most
    # one sequence from each example is inside the info queue and data_queue
    def _append_samples(self, size):
        if self.balanced:
            class_count = len(self.examples)
            # Size has to be N * number_of_classes, where N is an integer
            assert size % class_count == 0

            # Extract s samples per class
            s = size // class_count
            for i in range(s):
                # Get info about random example for each class
                examples = [self._get_random_example(class_label) for class_label in range(class_count)]
                if None in examples:
                    return

                # Put info about random examples to the queue
                for example in examples:
                    self.samples_count += 1
                    self.info_queue.put(example.get_info_and_advance())

        else:
            for _ in range(size):
                # Get a random example from any class
                example = self._get_random_example(class_label=None)
                if example is None:
                    return

                self.samples_count += 1
                self.info_queue.put(example.get_info_and_advance())

    def initialize_epoch(self, sequence_size, randomize):
        if self.data_type is not self.Train_Data and randomize:
            logger.warning('Are you sure you want to set randomize=True for non training data?')

        if self.continuous and self.epoch_initialized:
            logger.debug('Trying to initialize a new epoch (%s) but mode is set to continuous, skipping'
                         % self.data_type)
            return

        logger.info('Initialize new epoch (%s)' % self.data_type)
        # Read examples that we were unable to process
        for i in range(self.samples_count):
            self.data_queue.get()
            self.samples_count -= 1

        assert self.samples_count == 0

        for class_examples in self.examples:
            for example in class_examples:
                example.reset(sequence_size=sequence_size, randomize=randomize, new_epoch=True)
                example.state = self.state_initializer()

        # Populate info queue with some examples
        self._append_samples(5 * self.batch_size)

        self.state_needs_update = False
        self.epoch_initialized = True
        self._last_sequence_size = sequence_size
        self._last_randomize = randomize

    def set_states(self, keys, states):
        assert(len(keys) == len(states))

        for key, state in zip(keys, states):
            # If continuous training (no epochs) then we need to reset example if it was finished (done)
            if self.continuous and self.examples_dict[key].done == True:
                self.examples_dict[key].reset(self._last_sequence_size, self._last_randomize, new_epoch=False)
                self.examples_dict[key].state = self.state_initializer()
            else:
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
        context_arrays = []

        if self.samples_count < self.batch_size:
            if not self.allow_smaller_batch or self.samples_count == 0:
                raise SequenceDataReader.EpochDone
            else:
                batch_size = self.samples_count
        else:
            batch_size = self.batch_size

        for i in range(batch_size):
            data, time, label, example_id, context = self.data_queue.get()
            self.samples_count -= 1
            data_arrays.append(data)
            time_arrays.append(time)
            labels.append(label)
            ids.append(example_id)
            context_arrays.append(context)

        self.state_needs_update = True
        return ids, np.stack(data_arrays, axis=0), np.stack(time_arrays, axis=0), np.stack(labels, axis=0), \
               np.stack(context_arrays, axis=0)
