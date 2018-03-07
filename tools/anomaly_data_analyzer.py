from src.data_preparation.anomaly_data_generator import DataGenerator
import matplotlib.pyplot as plt
import glob
import os
import json


class AnomalyDataAnalyzer:
    def __init__(self, data_path, cache_path):
        self.data_path = data_path
        self.cache_path = cache_path

    def duration_histogram(self):
        file_paths = glob.glob(os.path.join(self.data_path, '**/*.edf'), recursive=True)
        print('Found file paths')
        edf_infos = [DataGenerator.FileInfo(file_path) for file_path in file_paths]
        print('Created EDF Info objects')
        d = [e.duration for e in edf_infos]

        print(max(d))
        print(min(d))
        print(sum(d)/len(d))

    # Not a clever and super fast solution but shoud work
    def analyze_age_gender(self):
        def load_data(folder):
            data = dict()
            info_files = os.listdir(os.path.join(self.cache_path, folder, 'info'))
            for info_file in info_files:
                with open(os.path.join(self.cache_path, folder, 'info', info_file)) as f:
                    info_dict = json.load(f)

                    key = (info_dict['age'], info_dict['gender'])

                    try:
                        data[key][info_dict['anomaly']] += 1
                    except KeyError:
                        try:
                            data[key][0] = 0
                            data[key][1] = 0
                            data[key][info_dict['anomaly']] += 1
                        except KeyError:
                            data[key] = {}
                            data[key][0] = 0
                            data[key][1] = 0
                            data[key][info_dict['anomaly']] += 1
            return data

        train_data = load_data('test')
        test_data = load_data('test')

        # Decision grid based on age
        keys_m = sorted([k for k in train_data.keys() if k[1] == 'M'], key=lambda k: k[0])
        keys_f = sorted([k for k in train_data.keys() if k[1] == 'F'], key=lambda k: k[0])

        def find_threshold(sorted_keys, train_data):
            ages, accuracies = [], []
            true_negs = []
            # Could be O(N) but is not
            for th in range(len(sorted_keys)):
                pred_normal_keys = sorted_keys[:th]
                pred_abnormal_keys = sorted_keys[th:]

                # Normal recordings classified as normal
                true_pos = sum([train_data[key][0] for key in pred_normal_keys])

                # Abnormal recordings classified as normal
                false_pos = sum([train_data[key][1] for key in pred_normal_keys])

                # Abnormal recordings classified as abnormal
                true_neg = sum([train_data[key][1] for key in pred_abnormal_keys])

                # Abnormal recordings classified as normal
                false_neg = sum([train_data[key][0] for key in pred_abnormal_keys])


                ages.append(sorted_keys[th][0])
                accuracies.append((true_pos + true_neg) / (true_pos+true_neg+false_pos+false_neg))
                true_negs.append(true_neg)

            return ages, accuracies

        fig, axes = plt.subplots(2, 1)

        ages, accuracies = find_threshold(keys_m, train_data)
        axes[0].set_title('Male')
        axes[0].bar(ages, accuracies)
        axes[0].set_ylim(bottom=0.35, top=0.65)

        ages, accuracies = find_threshold(keys_f, train_data)
        axes[1].set_title('Female')
        axes[1].bar(ages, accuracies)
        axes[1].set_ylim(bottom=0.3, top=0.7)

        plt.show()


if __name__ is '__main__':
    analyzer = AnomalyDataAnalyzer('/mhome/gemeinl/data/normal_abnormal', '/mhome/chrabasp/data/anomaly_14min_100hz')

    analyzer.analyze_age_gender()

    print('Done...')
#    analyzer.duration_histogram()
