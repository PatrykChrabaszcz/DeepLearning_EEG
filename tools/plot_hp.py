import json
import os
import matplotlib.pyplot as plt
import seaborn as sns


dir = "/mhome/chrabasp/results_bo_chrono_anomaly"


configs = []
results = []
with open(os.path.join(dir, 'configs.json')) as f:
    for line in f.readlines():
        configs.append(json.loads(line))

with open(os.path.join(dir, 'results.json')) as f:
    for line in f.readlines():
        results.append(json.loads(line))

l = []
hps = {}
for c, r in zip(configs, results):
    c_d = c[1]
    r_budget = r[1]
    r_info = r[3]['info']

    acc = r_info['X_acc_all_log_prob']
    loss = r[3]['loss']

    if r_budget == 1.0:
        l.append(loss)
        for hp_name, hp_value in c_d.items():
            try:
                hps[hp_name].append(hp_value)
            except KeyError:
                hps[hp_name] = [hp_value]

c = {'dropout': 'r',
     'l2_decay': 'b',
     'lr': 'g'}
for hp_name, hp_array in hps.items():
    print('Start plotting %s' % hp_name)
    if hp_name != 'dropout':
        plt.scatter(l, hp_array, c=c[hp_name])
        plt.title(hp_name)

plt.show()
