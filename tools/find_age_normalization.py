import click
import os
from src.utils import load_dict
import numpy as np


@click.command()
@click.option('--data_path', type=click.Path(exists=True))
def main(data_path):
    path = os.path.join(data_path, 'train', 'info')
    ages = []
    for p in os.listdir(path):
        info = load_dict(os.path.join(path, p))
        ages.append(info['age'])

    ages = np.array(ages)

    print(ages.mean())
    print(ages.std())


if __name__ == '__main__':
    main()