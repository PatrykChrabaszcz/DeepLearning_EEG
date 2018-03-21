from configparser import ConfigParser
import os
import json

path = "/mhome/chrabasp/EEG_Results/L2_Test/train_manager"

folders = sorted([f.path for f in os.scandir(path) if f.is_dir()])

values = {
    'Adam': {
        '256': [],
        '32': []
    },
    'AdamW': {
        '256': [],
        '32': []
    }
}

for folder in folders:
    try:
        config = ConfigParser()
        config.read(os.path.join(folder, 'config.ini'))

        optimizer = config.get('model_trainer', 'optimizer')
        decay = config.get('model_trainer', 'l2_decay')

        size = config.get('model', 'rnn_hidden_size')
        with open(os.path.join(folder, 'train_summarized_results.json')) as f:
            train_results = json.load(f)

        validation_folder = sorted([f.path for f in os.scandir(folder) if f.is_dir()])[0]

        with open(os.path.join(validation_folder, 'validation_summarized_results.json')) as f:
            validation_results = json.load(f)

        values[optimizer][size].append((decay, train_results['log_loss_all'], validation_results['log_loss_all'], size))


    except Exception:
        pass

print(sorted(values['Adam']['32'], key=lambda c: c[0]))
print(sorted(values['AdamW']['32'], key=lambda c: c[0]))