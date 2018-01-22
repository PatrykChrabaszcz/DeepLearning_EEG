import numpy as np
from src.utils import save_dict


class ClassificationMetrics:
    class ExampleStatistics:
        def __init__(self, example_id):
            self.example_id = example_id
            self._cnt = 0
            self._cnt_2nd = 0
            self._loss = 0
            self._loss_2nd = 0
            self._votes = [0, 0]
            self._weights = [0, 0]
            self._log_weights = [0, 0]

            self._votes_80p = [0, 0]
            self._votes_90p = [0, 0]

            # If the last prediction was correct
            self._last_correct = 0

        def append(self, output, labels):
            #print(output, labels)
            assert (len(output) == len(labels))

            m = np.max(output, axis=1, keepdims=True)
            x = output - m
            output_exp = np.exp(x)

            prob = output_exp / np.sum(output_exp, axis=1, keepdims=True)

            labels = np.expand_dims(labels, axis=1)
            correct_label_prob = np.sum(prob * np.hstack((1-labels, labels)), axis=1)
            bad_label_prob = 1 - correct_label_prob

            self._votes[0] += np.sum(correct_label_prob > 0.5)
            self._votes[1] += np.sum(correct_label_prob < 0.5)
            self._votes_80p[0] += np.sum(correct_label_prob > 0.8)
            self._votes_80p[1] += np.sum(correct_label_prob < 0.2)
            self._votes_90p[0] += np.sum(correct_label_prob > 0.9)
            self._votes_90p[1] += np.sum(correct_label_prob < 0.1)

            self._weights[0] += np.sum(correct_label_prob)
            self._weights[1] += np.sum(bad_label_prob)

            self._log_weights[0] += np.sum(np.log(correct_label_prob))
            self._log_weights[1] += np.sum(np.log(bad_label_prob))

            last_prob = correct_label_prob[-1]
            self._last_correct = last_prob > 0.5

            losses = -np.log(correct_label_prob)

            self._loss += np.sum(losses)
            self._loss_2nd += np.sum(losses[-200:])
            self._cnt += len(output)
            self._cnt_2nd += 200

        def loss(self):
            return self._loss / self._cnt

        def loss_2nd(self):
            return self._loss_2nd / self._cnt_2nd

        def majority_accuracy(self):
            return self._votes[0] > self._votes[1]

        def weighted_accuracy(self):
            return self._weights[0] > self._weights[1]

        def log_weighted_accuracy(self):
            return self._log_weights[0] > self._log_weights[1]

        def last_accuracy(self):
            return self._last_correct

        def accuracy_80p(self):
            return self._votes_80p[0] > self._votes_80p[1]

        def accuracy_90p(self):
            return self._votes_90p[0] > self._votes_90p[1]

    def __init__(self):
        self.results = {}
        self.loss = []
        self.examples = 0

    def append_results(self, ids, output, labels, loss, batch_size):
        self.loss.append(loss*batch_size)
        self.examples += batch_size

        assert len(ids) == len(output)
        assert len(ids) == len(labels)

        for example_id, o, l in zip(ids, output, labels):
            if example_id not in self.results:
                self.results[example_id] = ClassificationMetrics.ExampleStatistics(example_id)
            self.results[example_id].append(o, l)

    def get_summarized_results(self):
        output = {}

        losses = []
        losses_2nd = []
        majority_accuracy = []
        weighted_accuracy = []
        log_weighted_accuracy = []
        last_accuracy = []
        accuracy_80p = []
        accuracy_90p = []

        for i, example_statistics in self.results.items():
            losses.append(example_statistics.loss())
            losses_2nd.append(example_statistics.loss_2nd())
            majority_accuracy.append(example_statistics.majority_accuracy())
            weighted_accuracy.append(example_statistics.weighted_accuracy())
            log_weighted_accuracy.append(example_statistics.log_weighted_accuracy())
            last_accuracy.append(example_statistics.last_accuracy())
            accuracy_80p.append(example_statistics.accuracy_80p())
            accuracy_90p.append(example_statistics.accuracy_90p())
        output['loss'] = np.mean(losses)
        output['loss_2nd'] = np.mean(losses_2nd)
        output['majority_accuracy'] = np.sum(majority_accuracy) / len(majority_accuracy)
        output['weighted_accuracy'] = np.sum(weighted_accuracy) / len(weighted_accuracy)
        output['log_weighted_accuracy'] = np.sum(log_weighted_accuracy) / len(log_weighted_accuracy)
        output['last_accuracy'] = np.sum(last_accuracy) / len(last_accuracy)
        output['80p_accuracy'] = np.sum(accuracy_80p) / len(accuracy_80p)
        output['90p_accuracy'] = np.sum(accuracy_90p) / len(accuracy_90p)

        print('Loss original: %g' % (sum(self.loss)/self.examples))
        print('Number of examples (subsequences): %g' % self.examples)

        return output

    def save_detailed_output(self, path):
        res = self.get_summarized_results()

        res['detailed_res'] = {}
        detailed_res = res['detailed_res']
        for id, r in self.results.items():
            detailed_res[id] = {}
            r_dict = detailed_res[id]
            print('For id %s' % id)

            v_corr = r._votes[0]
            v_incorr = r._votes[1]
            print('Log weights %s' % v_corr)
            print('Log weights %s' % v_incorr)

            print(v_corr / (v_corr + v_incorr))
            r_dict['prob_of_correct'] = v_corr / (v_corr + v_incorr)

        save_dict(res, path)


class RegressionMetrics:
    class Example_Statistics:
        def __init__(self, example_id):
            self.example_id = example_id
            self._cnt = 0
            self._loss = 0

        def append(self, output, labels):
            output = output.flatten()
            labels = labels.flatten()
            assert(len(output) == len(labels))
            self._loss += np.sum(np.square(output-labels))
            self._cnt += len(output)

        def loss(self):
            return self._loss/self._cnt

    def __init__(self):
        self.results = {}
        self.loss = []
        self.examples = 0

    def append_results(self, ids, output, labels, loss, batch_size):
        pass
        self.loss.append(loss*batch_size)
        self.examples += batch_size

        assert len(ids) == len(output)
        assert len(ids) == len(labels)

        # for example_id, o, l in zip(ids, output, labels):
        #     if example_id not in self.results:
        #         self.results[example_id] = ClassificationMetrics.ExampleStatistics(example_id)
        #     self.results[example_id].append(o, l)

    def get_summarized_results(self):

        output = {}
        #
        #losses = []
        # losses_2nd = []
        # majority_accuracy = []
        # weighted_accuracy = []
        # log_weighted_accuracy = []
        # last_accuracy = []
        # accuracy_80p = []
        # accuracy_90p = []
        #
        # for i, example_statistics in self.results.items():
        #     losses.append(example_statistics.loss())
        #     losses_2nd.append(example_statistics.loss_2nd())
        #     majority_accuracy.append(example_statistics.majority_accuracy())
        #     weighted_accuracy.append(example_statistics.weighted_accuracy())
        #     log_weighted_accuracy.append(example_statistics.log_weighted_accuracy())
        #     last_accuracy.append(example_statistics.last_accuracy())
        #     accuracy_80p.append(example_statistics.accuracy_80p())
        #     accuracy_90p.append(example_statistics.accuracy_90p())
        #output['loss'] = np.mean(losses)
        # output['loss_2nd'] = np.mean(losses_2nd)
        # output['majority_accuracy'] = np.sum(majority_accuracy) / len(majority_accuracy)
        # output['weighted_accuracy'] = np.sum(weighted_accuracy) / len(weighted_accuracy)
        # output['log_weighted_accuracy'] = np.sum(log_weighted_accuracy) / len(log_weighted_accuracy)
        # output['last_accuracy'] = np.sum(last_accuracy) / len(last_accuracy)
        # output['80p_accuracy'] = np.sum(accuracy_80p) / len(accuracy_80p)
        # output['90p_accuracy'] = np.sum(accuracy_90p) / len(accuracy_90p)
        output['loss'] = (sum(self.loss)/self.examples)
        # eprint('Loss original: %g' % )
        # print('Number of examples (subsequences): %g' % self.examples)
        #
        return output

    def save_detailed_output(self, path):
        pass
        # res = self.get_summarized_results()
        #
        # res['detailed_res'] = {}
        # detailed_res = res['detailed_res']
        # for id, r in self.results.items():
        #     detailed_res[id] = {}
        #     r_dict = detailed_res[id]
        #     print('For id %s' % id)
        #
        #     v_corr = r._votes[0]
        #     v_incorr = r._votes[1]
        #     print('Log weights %s' % v_corr)
        #     print('Log weights %s' % v_incorr)
        #
        #     print(v_corr / (v_corr + v_incorr))
        #     r_dict['prob_of_correct'] = v_corr / (v_corr + v_incorr)
        #
        # save_dict(res, path)


def create_metrics(objective_type):
    if objective_type == 'classification':
        return ClassificationMetrics()
    elif objective_type == 'regression':
        return RegressionMetrics()
    else:
        raise  NotImplementedError('In create_metrics(...) objective_type=%s is not implemented' % objective_type)


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