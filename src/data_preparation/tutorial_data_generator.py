import logging
import click
import os
import numpy as np


log = logging.getLogger()


class DataGenerator:
    def __init__(self, output_path, recording_length=10000, num_recordings=100):
        self.output_path = output_path
        self.recording_length = recording_length
        self.num_recordings = num_recordings

    def prepare(self):
        os.makedirs(self.output_path, exist_ok=True)
        for i in range(self.num_recordings):
            data = np.random.choice([-1, 1], size=self.recording_length)
            np.save(os.path.join(self.output_path, '%d.npy' % i), data)


@click.command()
@click.option('--output_path', type=click.Path(), required=True)
@click.option('--recording_length', default=10000, help='How many timepoints each recording has')
@click.option('--num_recordings', default=100, help='How many examples to generate')
def main(output_path, recording_length, num_recordings):
    print('Settings:')
    print('Output Path: %s' % output_path)
    print('Length of each recording: %s' % recording_length)
    print('Number of recordings: %s' % num_recordings)

    data_generator = DataGenerator(output_path, recording_length, num_recordings)
    data_generator.prepare()


if __name__ == "__main__":
    main()
