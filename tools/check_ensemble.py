import os
import json

dir = '/mhome/chrabasp/results_single/cp'


results = []
for train_run in os.listdir(dir):
    try:
        file = os.path.join(dir, train_run, 'valid', 'detailed_results.json')
        with open(file, 'r') as f:
            results.append(json.load(f))
    except:
        print('Could not process results from the folder: %s' % train_run)


scores = {}
for r in results:
    for key in r.keys():
        if key not in scores.keys():
            scores[key] = {}
            scores[key]['good'] = 0
            scores[key]['all'] = 0

        if int(r[key]['X_acc_all_log_prob']) == 1:
            scores[key]['good'] += 1
        scores[key]['all'] += 1

print(scores)

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



