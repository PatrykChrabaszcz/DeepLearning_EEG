from src.data_reading.anomaly_data_reader import AnomalyDataReader
import os
import json


def read_statistics(data_path, data_type):
    train_info_dicts = AnomalyDataReader.load_info_dicts(data_path, data_type)

    stdv_list = []
    mean_list = []
    names = []
    labels = []
    for info_dict in train_info_dicts:
        stdv_list.append(info_dict['std'])
        mean_list.append(info_dict['mean'])
        names.append(info_dict['sequence_name'])
        labels.append(int(info_dict['anomaly']))

    return names, stdv_list, mean_list, labels


def read_predictions(results_folder):
    results = []
    for file in os.walk(results_folder):
        file = os.path.join(file[0], "validation_detailed_results.json")
        try:
            with open(file, 'r') as f:
                results.append(json.load(f))
        except FileNotFoundError:
            pass

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
            scores[key]['acc'] = scores[key]['good']/scores[key]['all']

    return scores