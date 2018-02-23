from src.utils import save_dict
import numpy as np
import json
import os


class ClassificationMetrics:
    class ExampleStatistics:
        """
        I would use metrics from a library but that implementation is super slow,
        Variable names here might be confusing, names with X indicate statistics over examples
        where each example has the same weight. For names without X longer examples might contribute
        more to the aggregated statistics

        """
        def __init__(self, example_id, labels_cnt):
            self.example_id = example_id
            self._cnt_all = 0
            self._log_loss_all = 0
            self._acc_all = 0
            self._cnt_end = 0
            self._log_loss_end = 0
            self._log_loss_end_last = 0
            self._acc_end = 0
            self._acc_end_last = 0

            self._log_prob_all_g = 0
            self._log_prob_all_a = 0

            self._log_prob_end_g = 0
            self._log_prob_end_a = 0

            self.labels_cnt = labels_cnt

        def append(self, output, labels):
            assert (len(output) == len(labels))

            labels_one_hot = np.zeros(shape=(len(labels), self.labels_cnt))
            for c in range(self.labels_cnt):
                labels_one_hot[labels == c, c] = 1

            output_labels = np.argmax(output, axis=1)
            output_exp = np.exp(output - np.max(output, axis=1, keepdims=True))
            output_prob = output_exp / np.sum(output_exp, axis=1, keepdims=True)
            eps = 1e-10
            output_prob = np.clip(output_prob, eps, 1 - eps)
            losses = -np.sum(labels_one_hot * np.log(output_prob), axis=1)
            accuracy = output_labels == labels

            log_weighted_prob = np.log(output_prob)
            log_weighted_prob_g = log_weighted_prob[np.arange(len(output)), labels]

            self._cnt_all += len(output)
            self._log_loss_all += np.sum(losses)
            self._acc_all += int(np.sum(accuracy))
            self._cnt_end += 1
            self._log_loss_end += losses[-1]
            self._log_loss_end_last = losses[-1]
            self._acc_end += int(accuracy[-1])
            self._acc_end_last = int(accuracy[-1])

            # g-good, a-all
            self._log_prob_all_g += np.sum(log_weighted_prob_g)
            self._log_prob_all_a += np.sum(log_weighted_prob)

            self._log_prob_end_g += log_weighted_prob_g[-1]
            self._log_prob_end_a += np.sum(log_weighted_prob[-1])

        def stats(self):
            res = {
                'acc_end': self._acc_end,
                'X_acc_end': self._acc_end_last,
                'log_loss_end': self._log_loss_end,
                'X_log_loss_end': self._log_loss_end_last,
                'cnt_end': self._cnt_end,

                'acc_all': self._acc_all,
                'log_loss_all': self._log_loss_all,
                'cnt_all': self._cnt_all,

                'X_acc_all_log_prob': 1. if self._log_prob_all_g > (
                            self._log_prob_all_a - self._log_prob_all_g) else 0.,
                'X_acc_ends_log_prob': 1. if self._log_prob_end_g > (
                            self._log_prob_end_a - self._log_prob_end_g) else 0.,

                'X_acc_ends': 1. if self._acc_end > self._cnt_end // 2 else 0.,
                'X_acc_all': 1. if self._acc_all > self._cnt_all // 2 else 0.
            }
            return res

        @staticmethod
        def average_stats(stats):
            res = {}

            example_cnt = len(stats)
            cnt_all = sum(s['cnt_all'] for s in stats)
            cnt_end = sum(s['cnt_end'] for s in stats)

            res['example_cnt'] = int(example_cnt)
            res['cnt_all'] = int(cnt_all)
            res['cnt_end'] = int(cnt_end)

            for key in stats[0].keys():
                if 'cnt' in key:
                    continue
                elif 'X' in key:
                    res[key] = sum(s[key] for s in stats) / example_cnt
                elif 'end' in key:
                    res[key] = sum(s[key] for s in stats) / cnt_end
                elif 'all' in key:
                    res[key] = sum(s[key] for s in stats) / cnt_all
                else:
                    raise NotImplementedError('Can not average over %s key ' % key)
            return res

    def __init__(self, name, output_size):
        self.output_size = output_size
        self.examples = {}
        self.loss = 0
        self.loss_cnt = 0

        self.recent_loss_array = [0] * 100
        self.recent_loss_bs_array = [0] * 100
        self.recent_loss_index = 0

        self.name = name

    def append_results(self, ids, output, labels, loss, batch_size):
        self.loss += loss * batch_size
        self.loss_cnt += batch_size

        self.recent_loss_array[self.recent_loss_index] = loss * batch_size
        self.recent_loss_bs_array[self.recent_loss_index] = batch_size
        self.recent_loss_index = (self.recent_loss_index + 1) % 100

        assert len(ids) == len(output)
        assert len(ids) == len(labels)

        for example_id, o, l in zip(ids, output, labels):
            if example_id not in self.examples:
                self.examples[example_id] = ClassificationMetrics.ExampleStatistics(example_id, self.output_size)
            self.examples[example_id].append(o, l)

    def get_summarized_results(self):
        stats = [v.stats() for (k, v) in self.examples.items()]
        res = self.ExampleStatistics.average_stats(stats)

        res['loss'] = self.loss/self.loss_cnt
        res['recent_loss'] = sum(self.recent_loss_array) / sum(self.recent_loss_bs_array)

        return res

    def get_current_loss(self):
        return sum(self.recent_loss_array)/sum(self.recent_loss_bs_array)

    def save(self, directory):
        os.makedirs(directory, exist_ok=True)

        summarized_res = self.get_summarized_results()

        detailed_res = {}
        for example_id, example in self.examples.items():
            detailed_res[example_id] = example.stats()

        with open(os.path.join(directory, '%s_summarized_results.json' % self.name), 'w') as f:
            json.dump(summarized_res, f, sort_keys=True, indent=2)

        with open(os.path.join(directory, '%s_detailed_results.json' % self.name), 'w') as f:
            json.dump(detailed_res, f, sort_keys=True, indent=2)


# TODO Write Regression Metrics
class RegressionMetrics:
    class ExampleStatistics:
        def __init__(self, example_id):
            self.example_id = example_id
            self._cnt = 0

            self._predictions = []
            self._labels = []

        def append(self, output, labels):
            assert output.shape == labels.shape
            self._predictions.append(output)
            self._labels.append(labels)
            self._cnt += len(output)

        def stats(self):
            predictions = np.concatenate(self._predictions)
            labels = np.concatenate(self._labels)

            loss = np.mean(np.square(predictions - labels))

            correlations = []
            for i in range(predictions.shape[1]):
                correlations.append(np.corrcoef(predictions[:, i], labels[:, i])[0][1])

            return loss, correlations

    def __init__(self):
        raise NotImplementedError('Regression metrics currently not implemented')
        # self.results = {}
        # self.loss = []
        # self.examples = 0
    #
    # def append_results(self, ids, output, labels, loss, batch_size):
    #     self.loss.append(loss*batch_size)
    #     self.examples += batch_size
    #
    #     assert len(ids) == len(output)
    #     assert len(ids) == len(labels)
    #
    #     for example_id, o, l in zip(ids, output, labels):
    #         if example_id not in self.results:
    #             self.results[example_id] = RegressionMetrics.ExampleStatistics(example_id)
    #         self.results[example_id].append(o, l)
    #
    # def get_summarized_results(self):
    #
    #     output = {}
    #
    #     for i, r in self.results.items():
    #         print(r.stats())
    #         break
    #         #stats = self.results


def create_metrics(name, objective_type, output_size):
    if 'CrossEntropy' in objective_type:
        return ClassificationMetrics(name, output_size)
    elif 'MeanSquaredError' in objective_type or 'L1Loss' in objective_type:
        return RegressionMetrics()
    else:
        raise NotImplementedError('In create_metrics(...) objective_type=%s is not implemented' % objective_type)


def average_metrics_results(results):
    res = {}
    for key in results[0]:
        res[key] = sum(r[key] for r in results)/len(results)

    return res

if __name__ == "__main__":
    outputs = np.array([[[15, 15], [10, 11]],
                        [[15, 15], [10, 13]],
                        [[15, 15], [1, -1]]])
    labels = np.array([[0, 0],
                       [0, 1],
                       [1, 0]])
    print(len(labels))
    ids = [0, 1, 2]

    metrics = ClassificationMetrics()

    metrics.append_results(ids, outputs, labels)
    r = metrics.finish_epoch()

    print(r)