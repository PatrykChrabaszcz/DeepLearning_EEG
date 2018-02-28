import matplotlib.pyplot as plt
import seaborn as sns
import click
import os

from src.data_preparation.anomaly_data_generator import DataGenerator
from src.data_reading.anomaly_data_reader import AnomalyDataReader


@click.command()
@click.option('--data_path', type=click.Path(exists=True), required=True)
@click.option('--output_dir', type=click.Path(), required=True)
@click.option('--data_type', type=str, required=True)
def main(data_path, output_dir, data_type):
    channels = DataGenerator.wanted_electrodes['EEG'] + DataGenerator.wanted_electrodes['EKG']

    train_info_dicts = AnomalyDataReader.load_info_dicts(data_path, data_type)

    stdv_list = []
    mean_list = []
    for info_dict in train_info_dicts:
        stdv_list.append(info_dict['std'])
        mean_list.append(info_dict['mean'])

    os.makedirs(output_dir, exist_ok=True)
    for channel_i, channel in enumerate(channels):
        stdv = [float(s[channel_i]) for s in stdv_list]
        mean = [float(m[channel_i]) for m in mean_list]

        plt.title(channel)
        sns.distplot(stdv, rug=True)
        plt.savefig(os.path.join(output_dir, channel + ' std'))
        plt.clf()

        plt.title(channel)
        sns.distplot(mean, rug=True)
        plt.savefig(os.path.join(output_dir, channel + ' mean'))
        plt.clf()


if __name__ == '__main__':
    main()
