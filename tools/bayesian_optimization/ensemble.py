import numpy as np
import json
import os


folder = '/mhome/chrabasp/EEG_Results/BO_Anomaly/train_manager'
scores = {}


for train_run_dir in sorted([f.path for f in os.scandir(folder) if f.is_dir()]):
    validation_dirs = [f.path for f in os.scandir(train_run_dir) if f.is_dir()]

    if len(validation_dirs) != 1:
        print('Expected one validation dir but found %d for %s' % (len(validation_dirs), train_run_dir))
        continue

    validation_dir = validation_dirs[0]

    detailed_results_file = os.path.join(validation_dir, 'validation_detailed_results.json')
    summarized_results_file = os.path.join(validation_dir, 'validation_summarized_results.json')

    try:
        with open(detailed_results_file) as f:
            detailed_results = json.load(f)

        with open(summarized_results_file) as f:
            summarized_results = json.load(f)
    except FileNotFoundError:
        print('No file available for the run %s, Skipping...' % train_run_dir)
        continue

    if float(summarized_results['X_acc_all_log_prob']) < 0.82:
        continue

    for key in detailed_results.keys():
        if key not in scores.keys():
            scores[key] = {}
            scores[key]['good'] = 0
            scores[key]['all'] = 0

        if int(detailed_results[key]['X_acc_all_log_prob']) == 1:
            scores[key]['good'] += 1
        scores[key]['all'] += 1

    print('Done for train run: %s' % train_run_dir)


print('Number of recordings %d' % len(scores.keys()))
good = 0
bad = 0

for s_name, s_score in scores.items():
    if s_score['good'] >= s_score['all'] / 2:
        good += 1
    else:
        bad += 1

print(good)
print(bad)
print(good/(good + bad))

print('Finished ...')


