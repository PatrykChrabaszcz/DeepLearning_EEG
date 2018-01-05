import numpy as np
from src.utils import Stats

class ClassificationMetrics:
    class Example_Statistics:
        def __init__(self, example_id):
            self.example_id = example_id
            self._cnt = 0
            self._loss = 0
            self._correct_votes = 0
            self._incorrect_votes = 0
            self._correct_weight = 0
            self._incorrect_weight = 0

        def append(self, output, labels):
            assert (len(output) == len(labels))

            r = np.arange(len(output))
            prob = np.exp(output[r, labels]) / (np.exp(output[r, 0]) + np.exp(output[r, 1]))

            self._correct_votes += np.sum(prob > 0.5)
            self._incorrect_votes += np.sum(prob < 0.5)

            self._correct_weight += np.sum(prob)
            self._incorrect_weight += len(prob) - np.sum(prob)

            losses = -np.log(prob)

            self._loss += np.sum(losses)
            self._cnt += len(output)

        def loss(self):
            return self._loss / self._cnt

        def majority_accuracy(self):
            return self._correct_votes > self._incorrect_votes

        def weighted_accuracy(self):
            return self._correct_weight > self._incorrect_weight

    def __init__(self):
        # Each sample (individual recording) has it's own element in this dictionary
        # For each sample there can be multiple predictions from different timestamps
        self.train_results = {}
        self.test_results = {}

    def append_results(self, ids, output, labels, train=True):
        results = self.train_results if train else self.test_results

        for example_id, o, l in zip(ids, output, labels):
            if example_id not in results:
                results[example_id] = ClassificationMetrics.Example_Statistics(example_id)

            results[example_id].append(o, l)

    def finish_epoch(self, train=True):
        output = {}
        results = self.train_results if train else self.test_results

        losses = []
        majority_accuracy = []
        weighted_accuracy = []

        for i, example_statistics in results.items():
            losses.append(example_statistics.loss())
            majority_accuracy.append(example_statistics.majority_accuracy())
            weighted_accuracy.append(example_statistics.weighted_accuracy())
        output['loss'] = np.mean(losses)
        output['majority_accuracy'] = np.sum(majority_accuracy) / len(majority_accuracy)
        output['weighted_accuracy'] = np.sum(weighted_accuracy) / len(majority_accuracy)

        results.clear()

        return output


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
        # Each sample (individual recording) has it's own element in this dictionary
        # For each sample there can be multiple predictions from different timestamps
        self.train_results = {}
        self.test_results = {}

    def append_results(self, ids, output, labels, train=True):
        results = self.train_results if train else self.test_results

        for example_id, o, l in zip(ids, output, labels):
            if example_id not in results:
                results[example_id] = RegressionMetrics.Example_Statistics(example_id)

            results[example_id].append(o, l)

    def finish_epoch(self, train=True):
        output = {}
        results = self.train_results if train else self.test_results

        losses = []
        for i, example_statistics in results.items():
            losses.append(example_statistics.loss())

        print('Loss %g' % np.mean(losses))

        output['loss'] = np.mean(losses)

        results.clear()

        return output