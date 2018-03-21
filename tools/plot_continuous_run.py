from configparser import ConfigParser
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json


def plot_experiment(path, plot_var, name_keys=(), ax=None, c=None):
    folders = sorted([f.path for f in os.scandir(path) if f.is_dir()])

    config = ConfigParser()
    config.read(os.path.join(path, 'config.ini'))


    name_parts = []
    for key in name_keys:
        name_parts.append(key[1] + ':' + config.get(*key))
    name = '__'.join(name_parts)

    p = []
    for folder in folders:
        with open(os.path.join(path, folder, 'validation_summarized_results.json')) as f:
            results = json.load(f)
            p.append(results[plot_var])

    dt = int(config.get('model_trainer', 'budget'))

    t = [dt*(i+1) for i in range(len(p))]
    sns.tsplot(p, time=t, condition=name, ax=ax, color=c)


if __name__ == '__main__':
    name_keys = (('model_trainer', 'objective_type'), ('model', 'rnn_num_layers'))
    plot_var = 'X_acc_all_log_prob'
    #plot_var = 'log_loss_all'
    main_folder = '/mhome/chrabasp/14_03_test_all_vs_last/train_manager/'
    folders = sorted([f.path for f in os.scandir(main_folder) if f.is_dir()])

    ax = plt.axes()
    colors = sns.color_palette(n_colors=4)
    for folder, c in zip(folders, colors):
        plot_experiment(folder, plot_var=plot_var, name_keys=name_keys, ax=ax, c=c)
    plt.show()
