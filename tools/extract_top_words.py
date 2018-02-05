import os
from src.data_preparation.anomaly_data_generator import DataGenerator
import glob
import re
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
import matplotlib
matplotlib.use('Qt5Agg')


data_path = '/mhome/gemeinl/data'
version = 'v1.1.2'


# Return File names for ["train", "eval"] X ["normal", "abnormal"]
def file_names(data_path, version, train=True, normal=True):
    mode = 'train' if train else 'eval'
    label = 'normal' if normal else 'abnormal'
    sub_path = 'normal_abnormal/{label}{version}/{version}/' \
               'edf/{mode}/{label}/'.format(mode=mode, label=label, version=version)
    path = os.path.join(data_path, sub_path)
    return DataGenerator.read_all_file_names(path, key=DataGenerator.Key.time_key, extension='.txt')


counters = []
for normal in [True, False]:
    files = file_names(data_path=data_path, version=version, normal=normal)

    regex = re.compile(r'.*s\d\d.txt')

    f_files = list(filter(regex.search, files))

    counter = Counter()
    for i, file in enumerate(f_files):
        print('Processing file %f' % (i/len(f_files)))
        with open(file) as f:
            counter.update(list(set(f.read().split())))

    counters.append(counter)
    print('Counter updated')

df_all = pd.DataFrame.from_dict(counters[0] + counters[1], orient='index').reset_index()
df_all.set_index('index', inplace=True)
df_all.columns = ['Total']

df_normal = pd.DataFrame.from_dict(counters[0], orient='index').reset_index()
df_normal.set_index('index', inplace=True)
df_normal.columns = ['Normal']

df = pd.concat([df_all, df_normal], axis=1)
df['Abnormal'] = df['Total'] - df['Normal']
df['words'] = df.index
df['ratio'] = df['Abnormal'] / df['Total']
df['ratio2'] = df['Abnormal'] / (df['Normal'] + 1)

df = df.sort_values(by='ratio', ascending=False)[:100]

print(df)


sns.set(style="whitegrid", font_scale=0.7)
f, ax = plt.subplots(figsize=(6, 15))

sns.set_color_codes("pastel")
sns.barplot(x='Total', y='words', data=df, label='Total', color='b')


sns.set_color_codes("muted")
sns.barplot(x='Normal', y='words', data=df, label='Normal', color='b')
ax.legend(ncol=2, loc="lower right", frameon=True)

plt.tick_params(axis='both', which='y', labelsize=4)
plt.tick_params(axis='both', which='y', labelsize=4)
plt.ylabel('xlabel', fontsize=8)
plt.interactive(False)
plt.show()



