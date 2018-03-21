import os
from configparser import ConfigParser
import json
import collections
import scipy.stats as ss


nested_dict = lambda: collections.defaultdict(nested_dict)


main_folder = '/mhome/chrabasp/EEG_Results/RandomConfigurations_TitanX'
folders = sorted([f.path for f in os.scandir(main_folder) if f.is_dir()])

results = nested_dict()

for folder in folders:
    print('Processing folder %s' % folder)
    folder = os.path.join(folder, 'train_manager')
    for train_run_folder in sorted([f.path for f in os.scandir(folder) if f.is_dir()]):
        configuration_file = os.path.join(train_run_folder, 'config.ini')
        configuration = ConfigParser()
        configuration.read(configuration_file)

        cv_n = configuration.get('data_reader', 'cv_n')
        cv_k = configuration.get('data_reader', 'cv_k')
        budget = configuration.get('model_trainer', 'budget')

        # Should be only one validation folder
        validation_run_folder = [f.path for f in os.scandir(train_run_folder) if f.is_dir()]
        assert len(validation_run_folder) == 1
        validation_run_folder = validation_run_folder[0]

        with open(os.path.join(validation_run_folder, 'validation_summarized_results.json')) as f:
            val_results = json.load(f)

        acc = val_results['X_acc_all_log_prob']

        results[folder][budget][cv_n][cv_k] = acc

grouped_results = dict(zip(['A1', 'A2', 'B1', 'B2', 'B3', 'C1', 'C2', 'D1'], [[] for i in range(8)]))

for folder in sorted(results.keys()):
    res = []
    for budget in ['20', '60']:
        for cv_n in ['3', '9']:
            acc_list = []
            for cv_k in [str(i) for i in range(int(cv_n))]:
                acc = results[folder][budget][cv_n][cv_k]
                acc_list.append(acc)
                # Configuration A
                if cv_k == '2' and cv_n == '3' and budget == '20':
                    grouped_results['A1'].append(acc)
                if cv_k == '8' and cv_n == '9' and budget == '20':
                    grouped_results['A2'].append(acc)

                if cv_k == '2' and cv_n == '3' and budget == '60':
                    grouped_results['B1'].append(acc)
                if cv_k == '8' and cv_n == '9' and budget == '60':
                    grouped_results['B2'].append(acc)

            mean_acc = sum(acc_list) / len(acc_list)
            if cv_n == '3' and budget == '20':
                grouped_results['B3'].append(mean_acc)

            if cv_n == '3' and budget == '60':
                grouped_results['C1'].append(mean_acc)
            if cv_n == '9' and budget == '20':
                grouped_results['C2'].append(mean_acc)

            if cv_n == '9' and budget == '60':
                grouped_results['D1'].append(mean_acc)


keys = sorted(grouped_results.keys())
grouped_ranks = {}
for key in keys:
    items = grouped_results[key]
    print('Accuracy for %s' % key)
    print(['%.2f' % elem for elem in items])

    # Compute ranks
    ranks = ss.rankdata(items).tolist()
    grouped_ranks[key] = ranks

    print('and ranks:')
    print(ranks)


corr_dict = grouped_ranks

# Compute correlations

for a in ['A1', 'A2']:
    for b in ['B1', 'B2', 'B3']:
        print('%s, %s correlation:' % (a, b))
        print('%.3f' % ss.pearsonr(corr_dict[a], corr_dict[b])[0])

for b in ['B1', 'B2', 'B3']:
    for c in ['C1', 'C2']:
        print('%s, %s correlation:' % (b, c))
        print('%.3f' % ss.pearsonr(corr_dict[b], corr_dict[c])[0])

for c in ['C1', 'C2']:
    for d in ['D1']:
        print('%s, %s correlation:' % (c, d))
        print('%.3f' % ss.pearsonr(corr_dict[c], corr_dict[d])[0])




