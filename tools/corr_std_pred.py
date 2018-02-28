import matplotlib.pyplot as plt
import seaborn as sns
from tools.utils import read_statistics, read_predictions
import click
import os
import numpy as np
from src.data_preparation.anomaly_data_generator import DataGenerator


@click.command()
@click.option('--data_path', type=click.Path(exists=True), required=True)
@click.option('--results_path', type=click.Path(exists=True), required=True)
@click.option('--output_dir', type=click.Path(), required=True)
@click.option('--data_type', type=str, required=True)
def main(data_path, results_path, output_dir, data_type):
    channels = DataGenerator.wanted_electrodes['EEG'] + DataGenerator.wanted_electrodes['EKG']

    names, stdv_list, mean_list, labels = read_statistics(data_path, data_type)

    # s_dict = {}
    # for n, s in zip(names, stdv_list):
    #     s_dict[n] = s[8]
    #
    # l_dict = {}
    # for n, l in zip(names, labels):
    #     l_dict[n] = l
    #
    # predictions = read_predictions(results_path)
    # x, y, l = [], [], []
    # for sequence_name in predictions.keys():
    #     x.append(predictions[sequence_name]['acc'])
    #     y.append(s_dict[sequence_name])
    #     l.append(l_dict[sequence_name])
    #
    # plt.scatter(x, y, c=labels, s=5)
    #
    # plt.show()

    # Find max

    for i in range(22):
        s_list = [float(s[i]) for s in stdv_list]
        print(s_list)
        best_args = np.array(s_list).argsort()[::-1]

        print('Names:')
        for j in range(5):
            print(names[best_args[j]])


if __name__ == '__main__':
    main()
