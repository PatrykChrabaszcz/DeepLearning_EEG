from src.data_generator import DataGenerator
import click


@click.command()
@click.option('--data_path', type=click.Path(exists=True), required=True)
@click.option('--cache_path', type=click.Path(), required=True)
@click.option('--secs_to_cut', default=60, help='How many seconds are removed from the beginning and end of recording.')
@click.option('--sampling_freq', default=100.0, help='Signal will be preprocessed to the desired frequency.')
@click.option('--duration_min', default=5, help='Duration of the recording.')
@click.option('--val_split_factor', default=0.8, help='How much data is used for training (0.8: 80%)')
def main(data_path, cache_path, secs_to_cut, sampling_freq, duration_min, val_split_factor):
    print('Settings:')
    print('Data path: %s' % data_path)
    print('Cache path: %s' % cache_path)
    print('Seconds to cut (beginning and end): %d' % secs_to_cut)
    print('Sampling frequency: %d' % sampling_freq)
    print('Duration of the recording: %d' % duration_min)
    print('Train/Validation factor: %f' % val_split_factor)

    data_generator = DataGenerator(data_path, cache_path, secs_to_cut, sampling_freq, duration_min)
    data_generator.prepare(0.8)

if __name__ == "__main__":
    main()
