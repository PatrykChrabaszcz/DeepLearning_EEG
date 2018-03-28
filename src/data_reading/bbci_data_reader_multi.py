from src.data_reading.data_reader import SequenceDataReader
from braindecode.datasets.bbci import BBCIDataset
from braindecode.datautil.trial_segment import create_signal_target_from_raw_mne
from collections import OrderedDict
import logging
import os
import numpy as np


logger = logging.getLogger(__name__)


class BBCIDataReaderMulti(SequenceDataReader):
    class ExampleInfo(SequenceDataReader.SequenceExampleInfo):
        """
        Since those examples are quite short, it is better for validation to use full sequence size instead of
        dividing the data into chunks. Keep that in mind.
        """
        def __init__(self, example_id, random_mode, offset_size, data, label, context):
            super().__init__(example_id, random_mode=random_mode, offset_size=offset_size)

            self.label = label
            self.data = np.transpose(data)
            # Now data has shape time x features

            # No context for now for this data
            self.context = np.array([context]).astype(np.float32)

        def read_data(self, serialized):
            index, sequence_size = serialized

            # Time might be needed for some advanced models like PhasedLSTM
            time = np.reshape(np.arange(index, index + sequence_size), newshape=[sequence_size, 1])
            data = self.data[index: index+sequence_size]
            label = np.array([self.label] * sequence_size)

            return data, time, label, self.example_id, self.context

        def get_length(self):
            return self.data.shape[0]

    @staticmethod
    def add_arguments(parser):
        SequenceDataReader.add_arguments(parser)
        parser.add_argument("subject_names", type=str, help="TODO")
        parser.add_argument("load_sensor_names", type=str,
                            help="Sensor names, provide without spaces and separate using comma.")
        parser.add_argument("segment_ival_ms_start", type=int, default=-500)
        parser.add_argument("segment_ival_ms_end", type=int, default=4000)

    def _initialize(self, subject_name, load_sensor_names, segment_ival_ms_start, segment_ival_ms_end, **kwargs):
        folder = "BBCI-without-last-runs" if self.data_type != SequenceDataReader.Test_Data else "BBCI-only-last-runs"


        self.filename = os.path.join(self.data_path, folder, "%s.BBCI.mat" % subject_name)
        self.load_sensor_names = load_sensor_names.split(',')
        self.segment_ival_ms = [segment_ival_ms_start, segment_ival_ms_end]

    def _create_examples(self):
        cnt = BBCIDataset(self.filename, load_sensor_names=self.load_sensor_names).load()

        name_to_code = OrderedDict([('Right', 1), ('Left', 2), ('Rest', 3), ('Feet', 4)])

        data = create_signal_target_from_raw_mne(cnt, name_to_code, self.segment_ival_ms)
        data_list = [(d, l) for d, l in zip(data.X, data.y)]
        data_list = self.cv_split(data_list)

        # Create examples for 4 classes
        for label in range(4):
            class_data_list = [data for data in data_list if data[1] == label]

            self.examples.append([BBCIDataReader.ExampleInfo(example_id=(label, j), random_mode=self.random_mode,
                                                             offset_size=self.offset_size, label=label, data=data)
                                  for (j, (data, label)) in enumerate(class_data_list)])


    @staticmethod
    # Has to be a static method, context_size is required when creating the model,
    # DataReader can't be instantiated properly before the model is created
    def context_size(**kwargs):
        return 0

    @staticmethod
    def input_size(**kwargs):
        return 4

    @staticmethod
    def output_size(**kwargs):
        return 4


if __name__ == '__main__':
    from src.utils import setup_logging
    setup_logging('/tmp', logging.DEBUG)
    data_reader = BBCIDataReader(data_path='/home/schirrmr/data/',
                                 readers_count=1,
                                 batch_size=64,
                                 validation_batch_size=0,
                                 sequence_size=1125,
                                 validation_sequence_size=4500,
                                 balanced=1,
                                 random_mode=2,
                                 continuous=1,
                                 limit_examples=0,
                                 limit_duration=0,
                                 forget_state=1,
                                 train_on_full=0,
                                 cv_n=3,
                                 cv_k=2,
                                 force_parameters=0,
                                 offset_size=0,
                                 state_initializer=lambda: None,
                                 data_type=SequenceDataReader.Train_Data,
                                 allow_smaller_batch=0,
                                 subject_name="BhNoMoSc1S001R01_ds10_1-12",
                                 load_sensor_names='C3,CPz,C4',
                                 segment_ival_ms_start=-500,
                                 segment_ival_ms_end=4000)

    sequence_size = 1125
    try:
        data_reader.start_readers()
        data_reader.initialize_epoch(sequence_size)

        ids, batch, time, labels, contexts = data_reader.get_batch()
    except:
        raise
    finally:
        data_reader.stop_readers()



