from src.dl_tensorflow.model import SimpleRNN
#import unittest
from src.data_reading.finger_data_reader import FingerDataReader


class TestFingerDataReader:
    def test_data_reading(self):
        experiment_options = {
            'subject': 1,
            'split': 0.8
        }

        def state_initializer():
            return None

        dr = FingerDataReader(data_path='/mhome/chrabasp/BCICIV_4_mat',
                              experiment_options=experiment_options,
                              state_initializer=(lambda: None),
                              readers_count=1,
                              offset_size=0,
                              data_type=FingerDataReader.Train_Data,
                              batch_size=64,
                              balanced=False,
                              allow_smaller_batch=True,
                              continuous=False)

        dr.start_readers()
        dr.initialize_epoch(sequence_size=100, randomize=False)

        ids, batch, time, labels, contexts = dr.get_batch()

        print(time)

        dr.set_states(ids, [None] * len(ids))


if __name__ == '__main__':
    TestFingerDataReader().test_data_reading()

    #unittest.main()
