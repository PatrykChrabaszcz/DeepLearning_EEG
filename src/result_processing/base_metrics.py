import json
import os


class BaseMetrics:
    """
    Base class for metric computation. We consider the dataset as a set of examples, those examples might be represented
    as EEG recordings, MNIST images etc. This class will instantiate an object of ExampleClass type for each of those
    examples and will use this object to compute statistics/metrics for a given example.
    After each train/validation/test iteration ModelTrainer will call append_results function which in turn will split
    and delegate all results to corresponding ExampleClass objects. Those objects will compute and store all necessary
    information. At the end of the run we can call get_summarized_results() or save() to store the results in a json
    format.
    Note:
        Do not use this class directly! Use provided subclasses.

        Different metrics (implemented by different ExampleClass classes) might be good for different datasets.
        For example if you use metrics that compute output probabilities for each class on the text dataset that has
        hundreds or thousands of classes then metric computation might occupy 90% of your training time. In that
        cases consider using only simple Metric classes like SimpleLossMetric

    """
    def __init__(self, ExampleClass, name, output_size):
        self.ExampleClass = ExampleClass
        self.output_size = output_size
        self.examples = {}
        self.loss = 0
        self.loss_cnt = 0

        self.recent_loss_array = [0] * 100
        self.recent_loss_bs_array = [0] * 100
        self.recent_loss_index = 0

        self.name = name

    def append_results(self, ids, output, labels, loss):
        """
        Splits results from the minibatch and delegates metric computation to the proper example object.
        In addition stores information that is used to compute the loss from the last 100 iterations.
        Args:
            ids: ID values that will identify each example in the minibatch
            output: Output tensor as returned from the network
            labels: True targets
            loss: Loss value as computed by the network optimizer
        """

        batch_size = len(ids)
        self.loss += loss * batch_size
        self.loss_cnt += batch_size

        self.recent_loss_array[self.recent_loss_index] = loss * batch_size
        self.recent_loss_bs_array[self.recent_loss_index] = batch_size
        self.recent_loss_index = (self.recent_loss_index + 1) % 100

        assert len(ids) == len(output)
        assert len(ids) == len(labels)

        for example_id, o, l in zip(ids, output, labels):
            if example_id not in self.examples:
                self.examples[example_id] = self.ExampleClass(example_id, self.output_size)
            self.examples[example_id].append(o, l)

    def get_summarized_results(self):
        """
        Returns a dictionary with summarized results from all examples.
        Those results will be used by the worker to respond to the  Architecture Search Optimizer.
        """
        stats = [v.stats() for (k, v) in self.examples.items()]
        res = self.ExampleClass.average_stats(stats)

        res['loss'] = self.loss/self.loss_cnt
        res['recent_loss'] = sum(self.recent_loss_array) / sum(self.recent_loss_bs_array)

        return res

    def get_current_loss(self):
        """
        Returns an average loss from the last 100 iterations.
        """
        return sum(self.recent_loss_array)/sum(self.recent_loss_bs_array)

    def save(self, directory):
        """
        Saves both summarized results and results for each example using json files.
        Args:
         directory: Folder used to save the results. If not present then will try to create it.
        """
        os.makedirs(directory, exist_ok=True)

        summarized_res = self.get_summarized_results()

        detailed_res = {}
        for example_id, example in self.examples.items():
            detailed_res[example_id] = example.stats()

        with open(os.path.join(directory, '%s_summarized_results.json' % self.name), 'w') as f:
            json.dump(summarized_res, f, sort_keys=True, indent=2)

        with open(os.path.join(directory, '%s_detailed_results.json' % self.name), 'w') as f:
            json.dump(detailed_res, f, sort_keys=True, indent=2)

    @staticmethod
    def average_metrics_results(results):
        """
        One setting might consist of multiple CV folds, we want to be able to summarize summarized
        results from multiple train runs. This function takes an array with results and computes a mean for each of the
        keys. This simple implementation is sufficient for now.
        Args:
            results: A list with dictionaries as returned by the get_summarized_results() function.
        :return:
        """
        res = {}
        for key in results[0]:
            res[key] = sum(r[key] for r in results)/len(results)
        return res
