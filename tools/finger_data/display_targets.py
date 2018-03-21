from scipy.io import loadmat
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

subject_id = 1
finger_id = 0
file = '/mhome/chrabasp/data/finger/sub%d_comp.mat' % subject_id

data_dict = loadmat(file)

data = data_dict['train_data'].astype(np.float32)

# Remove 4th finger
labels = data_dict['train_dg'].astype(np.float32)[:320000, finger_id].flatten()

labels = np.clip(labels, -0.99, 100)
labels_l = np.log(1 + labels)
plt.hist(labels_l, bins=1000)
#sns.tsplot(labels)
plt.show()
