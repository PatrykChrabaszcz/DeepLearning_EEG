import json
import re


lukas_file = 'results/lukas_file.txt'
patryk_file = 'results/Validation.json'


def load_lukas_predictions():
    results = {}
    with open(lukas_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            match = re.findall('(\d{8}_s\d\d_a\d\d).edf\s+(\d)\s+(\d)', line)

            if len(match) == 1:
                [name, prediciton, label] = match[0]
                results[name] = prediciton == label
                print(name)
    return results


def load_patryk_predictions():
    with open(patryk_file, 'r') as f:
        results = json.load(f)['detailed_res']

    return results


if __name__ == '__main__':
    l_p = load_lukas_predictions()
    p_p = load_patryk_predictions()

    both_correct = 0
    both_wrong = 0
    l_correct = 0
    p_correct = 0
    for key, value in p_p.items():
        correct_p = value['prob_of_correct'] > 0.5
        correct_l = l_p[key]

        if correct_p == True and correct_l == True:
            both_correct += 1
        elif correct_l == False and correct_p == False:
            both_wrong += 1
        elif correct_l == True and correct_p == False:
            l_correct += 1
        elif correct_l == False and correct_p == True:
            p_correct += 1

    print('Number of samples %d' % len(p_p.keys()))

    print('Both correct %d' % both_correct)
    print('Both wrong %d' % both_wrong)
    print('RNN correct, AutoML wrong %d' % p_correct)
    print('RNN wrong, AutoML correct %d' % l_correct)
