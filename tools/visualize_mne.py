import os.path as op
import numpy as np

import mne

data_path = "/mhome/gemeinl/data/normal_abnormal/abnormalv1.1.2/v1.1.2/edf/train/abnormal/005/00000879/s02_2013_01_01/00000879_s02_a04.edf"
data_path_fif = "/mhome/chrabasp/data/anomaly_10min_not_clipped/train/data/2013_01_01_00000879_s02_a04_Age_29_Gender_M_raw.fif"
data_path2_fif = "/mhome/chrabasp/data/anomaly_10min_not_clipped/train/data/2013_01_01_00001863_s01_a00_Age_32_Gender_F_raw.fif"
raw = mne.io.read_raw_fif(data_path_fif, preload=True)

# cnt = raw.pick_channels(['EEG P3-REF'])
#
# data = np.array(cnt.get_data() * 1e6).astype(np.float32)
# print(data)
#raw = mne.io.read_raw_edf(data_path, preload=True)
#raw.set_eeg_reference('average', projection=True)  # set EEG average reference
#events = mne.read_events(data_path)
raw.plot(block=True)
