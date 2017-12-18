import resampy
import numpy as np
import logging
from src.utils import Stats
import glob
import re
import os
import mne


log = logging.getLogger()


class DataGenerator:
    """
    Applies preprocessing to the data.
    Extracts Train/Validation and Test datasets.
    Handles different channel order for different files
    Cache data
    """
    wanted_electrodes = {
        'EEG': ['EEG A1-REF', 'EEG A2-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG CZ-REF', 'EEG F3-REF', 'EEG F4-REF',
                'EEG F7-REF', 'EEG F8-REF', 'EEG FP1-REF', 'EEG FP2-REF', 'EEG FZ-REF', 'EEG O1-REF', 'EEG O2-REF',
                'EEG P3-REF', 'EEG P4-REF', 'EEG PZ-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF'],
        'EKG1': ['EEG EKG1-REF'],
        'EKG': ['ECG EKG-REF']

    }

    # More like a namespace than class
    class Key:
        @staticmethod
        def session_key(file_name):
            return re.findall(r'(s\d{2})', file_name)

        @staticmethod
        def natural_key(file_name):
            key = [int(token) if token.isdigit() else None for token in re.split(r'(\d+)', file_name)]
            return key

        @staticmethod
        def time_key(file_name):
            splits = file_name.split('/')
            [date] = re.findall(r'(\d{4}_\d{2}_\d{2})', splits[-2])
            date_id = [int(token) for token in date.split('_')]
            recording_id = DataGenerator.Key.natural_key(splits[-1])
            session_id = DataGenerator.Key.session_key(splits[-2])
            return date_id + session_id + recording_id

    class FileInfo:
        # Can throw a ValueError exception if file is wrongly formatted
        def __init__(self, file_path, preload=False):
            """ read info from the edf file without loading the data. loading data is done in multiprocessing since it takes
            some time. getting info is done before because some files had corrupted headers or weird sampling frequencies
            that caused the multiprocessing workers to crash. therefore get and check e.g. sampling frequency and duration
            beforehand
            :param file_path: path of the recording file
            :return: file name, sampling frequency, number of samples, number of signals, signal names, duration of the rec
            """
            self.edf_file = None
            self.sampling_frequency = None
            self.n_samples = None
            self.n_signals = None
            self.signal_names = None
            self.duration = None

            edf_file = mne.io.read_raw_edf(file_path, preload=preload, verbose='error')

            # Some recordings have a very weird sampling frequency. Check twice before skipping the file
            sampling_frequency = int(edf_file.info['sfreq'])
            if sampling_frequency < 10:
                sampling_frequency = 1 / (edf_file.times[1] - edf_file.times[0])
                if sampling_frequency < 10:
                    self.sampling_frequency = sampling_frequency
                    return

            self.edf_file = edf_file
            self.sampling_frequency = sampling_frequency
            self.n_samples = edf_file.n_times
            self.signal_names = edf_file.ch_names
            self.n_signals = len(self.signal_names)
            # Some weird sampling frequencies are at 1 hz or below, which results in division by zero
            self.duration = self.n_samples / max(sampling_frequency, 1)

    class RunningNormStats:
        def __init__(self, channels_cnt, factor=1000.0):
            self.factor = factor
            self.cnt = np.float64(0)
            self.sum = np.zeros([channels_cnt, 1], dtype=np.float64)
            self.sum_square = np.zeros([channels_cnt, 1], dtype=np.float64)

        def append_data(self, data):
            self.cnt += data.shape[1]
            self.sum +=np.sum(data, axis=1, keepdims=True)
            self.sum_square += np.sum(np.square(data), axis=1, keepdims=True)

        @property
        def mean(self):
            return self.sum / (self.cnt)

        @property
        def stdv(self):
            return np.sqrt(np.maximum((self.sum_square / self.cnt - np.square(self.mean)), 1e-15))

    def __init__(self, data_path, cache_path, secs_to_cut, sampling_freq, duration_min, version='v1.1.2'):
        """
        :param data_path:
            Path to the original data
        :param cache_path:
            Path to the derived data (User has to make sure that this directory is empty when new options are used)
        :param version:
            Dataset version
        """
        self.data_path = data_path
        self.cache_path = cache_path
        self.version = version

        self.secs_to_cut_at_start_end = secs_to_cut
        self.sampling_freq = sampling_freq
        self.duration_recording_mins = duration_min

    @staticmethod
    def read_all_file_names(path, extension='.edf', key=Key.time_key):
        file_paths = glob.glob(os.path.join(path, '**/*' + extension), recursive=True)
        return sorted(file_paths, key=key)

    # Return File names for ["train", "eval"] X ["normal", "abnormal"]
    def _file_names(self, train=True, normal=True):
        mode = 'train' if train else 'eval'
        label = 'normal' if normal else 'abnormal'
        sub_path = 'normal_abnormal/{label}{version}/{version}/' \
                   'edf/{mode}/{label}/'.format(mode=mode, label=label, version=self.version)
        path = os.path.join(self.data_path, sub_path)
        return self.read_all_file_names(path, key=DataGenerator.Key.time_key)

    # Get all file names for train and test data, sorted according to the time key
    def _get_all_sorted_file_names_and_labels(self, train=True):
        normal_file_names = self._file_names(train=train, normal=True)
        abnormal_file_names = self._file_names(train=train, normal=False)

        all_file_names = normal_file_names + abnormal_file_names
        all_file_names = sorted(all_file_names, key=DataGenerator.Key.time_key)

        abnormal_counts = [file_name.count('abnormal') for file_name in all_file_names]
        assert set(abnormal_counts) == {1, 3}
        labels = np.array(abnormal_counts) == 3
        labels = labels.astype(np.int64)

        return all_file_names, labels

    def _load_file(self, file_name, preprocessing_functions, sensor_types=('EEG',)):
        wanted_electrodes = []
        for sensor_type in sensor_types:
            wanted_electrodes.extend(DataGenerator.wanted_electrodes[sensor_type])

        # This guy can throw an exception (TODO:See)
        file_info = DataGenerator.FileInfo(file_name, preload=True)

        log.info("Load file %s" % file_name)
        cnt = file_info.edf_file.pick_channels(wanted_electrodes)

        if not np.array_equal(cnt.ch_names, wanted_electrodes):
            raise RuntimeError('Not all channels available')

        data = cnt.get_data().astype(np.float32)
        fs = cnt.info['sfreq']

        if preprocessing_functions is not None:
            for preprocessing_function in preprocessing_functions:
                log.info(preprocessing_function)

                data, fs = preprocessing_function(data, fs)
                assert (data.dtype == np.float32) and (type(fs) == float), (data.dtype, type(fs))

        return data

    def default_preprocessing_functions(self):
        preprocessing_functions = []

        preprocessing_functions.append(lambda data, fs: (data[:, int(self.secs_to_cut_at_start_end * fs):-int(
            self.secs_to_cut_at_start_end * fs)], fs))

        #preprocessing_functions.append(lambda data, fs: (data[:, :int(self.duration_recording_mins * 60 * fs)], fs))

        preprocessing_functions.append(lambda data, fs: (resampy.resample(data, fs, self.sampling_freq, axis=1,
                                                                          filter='kaiser_fast'), self.sampling_freq))

        return preprocessing_functions

    def prepare(self, split_factor):
        train_files, train_labels = self._get_all_sorted_file_names_and_labels(train=True)
        assert len(train_files) == len(train_labels) and len(train_files) != 0

        test_files, test_labels = self._get_all_sorted_file_names_and_labels(train=False)
        assert len(test_files) == len(test_labels) and len(test_files) != 0

        # Find out split index for train and validation
        split_index = int(len(train_files) * split_factor)

        val_files = train_files[split_index:]
        val_labels = train_labels[split_index:]

        train_files = train_files[:split_index]
        train_labels = train_labels[:split_index]

        # Find out normalization statistics:
        preprocessing_functions = self.default_preprocessing_functions()
        norm_stats = None
        for i, train_file in enumerate(train_files):
            try:
                data = self._load_file(train_file, preprocessing_functions, sensor_types=('EEG', 'EKG1'))
            except RuntimeError:
                data = self._load_file(train_file, preprocessing_functions, sensor_types=('EEG', 'EKG'))
            norm_stats = DataGenerator.RunningNormStats(channels_cnt=data.shape[0]) if norm_stats is None else norm_stats
            norm_stats.append_data(data)

            print('Find normalization, Progress %g' % ((i+1) / len(train_files)))

        mean = norm_stats.mean.astype(dtype=np.float32)
        stdv = norm_stats.stdv.astype(dtype=np.float32)

        preprocessing_functions.append(lambda data, fs: ((data-mean)/stdv, fs))

        ch_names = DataGenerator.wanted_electrodes['EEG'] + DataGenerator.wanted_electrodes['EKG']
        for split_type, split_files, split_labels in zip(['train', 'validation', 'test'],
                                                         [train_files, val_files, test_files],
                                                         [train_labels, val_labels, test_labels]):

            normal_path = os.path.join(self.cache_path, split_type, 'normal')
            abnormal_path = os.path.join(self.cache_path, split_type, 'abnormal')

            for i, (file, label) in enumerate(zip(split_files, split_labels)):
                output_path = normal_path if label == 0 else abnormal_path
                os.makedirs(output_path, exist_ok=True)
                output_path = os.path.join(output_path, str(i) + '_raw.fif')

                try:
                    data = self._load_file(file, preprocessing_functions, sensor_types=('EEG', 'EKG1'))
                except RuntimeError:
                    data = self._load_file(file, preprocessing_functions, sensor_types=('EEG', 'EKG'))

                info = mne.create_info(ch_names, sfreq=self.sampling_freq)
                fif_array = mne.io.RawArray(data, info)
                fif_array.save(output_path)

                print('Split Type: %s, Progress: %g' % (split_type, (i+1)/len(split_files)))


if __name__ == '__main__':
    with Stats():
        dataset = DataGenerator(data_path='/mhome/gemeinl/data', cache_path='/mhome/chrabasp/data_tmp')

    dataset.prepare(split_factor=0.8)


