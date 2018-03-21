from src.result_processing.base_metrics import BaseMetrics
import numpy as np


class Example:
    """
    Class used to compute statistics for single recording. If recording is long it might be divided into many
    chunks, data chunks might be even overlapping for train data if random_mode = 2 or if continuous = 1.
    Note: The way we compute accuracy using the sum of log probabilities over the whole recording works only
    because we have one label per recording. Take into account that you will not be able to use this
    class for the dataset that does not fulfil this requirement.
    """

    def __init__(self, example_id, labels_cnt):
        # Id - recording name makes it possible to track back which recording is reported
        self.example_id = example_id
        self.labels_cnt = labels_cnt

        # Has to be deduced
        self.true_label = None

        # Number of timepoints for which prediction was made
        self._cnt_all = 0
        # Number of data chunks, some statistics will only use the last timepoint from RNN predictions.
        # It could be a better prediction since RNN have seen more data.
        self._cnt_end = 0

        # Number of times given class was predicted. We consider all timepoints
        self._acc_all = np.zeros([1, labels_cnt])
        # Number of times given class was predicted. We consider only the timepoints at the end of each data chunk.
        self._acc_end = np.zeros([1, labels_cnt])

        # Sum of log probabilities for each class. We consider all timepoints
        self._log_prob_all = np.zeros([1, labels_cnt])
        # Sum of log probabilities for each class. We consider only the timepoints at the end of each data chunk.
        self._log_prob_end = np.zeros([1, labels_cnt])

        # 1 if last timepoint prediction from the last seen data chunk was correct, 0 if it was not correct
        self._acc_end_last = 0
        # Log probability for the correct class of the last data chunk.
        # Makes sense for validation pass if sequence_size was smaller than
        # recording size and data had to be divided into chunks.
        # For validation this will be the same as log_loss_end if sequence_size == recording_size
        self._log_prob_end_last = 0

    # Output should have shape [timepoints x labels_cnt]
    def append(self, output, labels):
        """
        Each time a new minibatch is processed by the rnn, all the samples from minibatch will be split and this
        function will be called for each of the samples. It simply updates the state that is used at the end
        to compute accumulated loss and accuracy.
        """
        assert len(output) == len(labels), 'Size mismatch for outputs and labels'
        assert output.shape[1] == self.labels_cnt, \
            'Number of labels %d not consistent with the network output %d' % (self.labels_cnt, output.shape[0])

        self.true_label = labels[0] if self.true_label is None else self.true_label
        assert all(x == self.true_label for x in labels), 'All labels should have the same value'

        # Number of timepoints in this data chunk
        num_timepoints = len(output)
        self.true_label = labels[0]

        # Compute probability of each label
        output_exp = np.exp(output - np.max(output, axis=1, keepdims=True))
        predicted_prob = output_exp / np.sum(output_exp, axis=1, keepdims=True)

        # We will take log prob and we want to somehow reduce the influence of predictions that have close to
        # 0 probability, we say that each class has at least 1% probability (Arbitrary decision)
        # Note that output_prob after that operation is not guaranteed to sum up to 100%.
        predicted_prob = np.clip(predicted_prob, 0.01, 1)

        # We compute log probability, better numerical properties for summing up log probabilities than for
        # multiplying probabilities.
        predicted_log_prob = np.log(predicted_prob)

        # Update internal state for this example
        self._cnt_all += num_timepoints
        self._log_prob_all += np.sum(predicted_log_prob, axis=0)
        self._acc_all += np.eye(self.labels_cnt)[predicted_log_prob.argmax(axis=1)].sum(axis=0)

        self._cnt_end += 1
        self._log_prob_end += predicted_log_prob[-1]
        self._acc_end += np.eye(self.labels_cnt)[predicted_log_prob[[-1], :].argmax(axis=1)]

        self._log_prob_end_last = predicted_log_prob[[-1], :]
        self._acc_end_last = predicted_log_prob[[-1], :].argmax(axis=1) == self.true_label

    def stats(self):
        res = {
            # Number of timepoints seen for this example
            'cnt_all': self._cnt_all,
            # Number of data chunks seen for this example
            'cnt_end': self._cnt_end,

            # Average log likelihood loss for the correct class using all timepoints
            'log_loss_all': float(-self._log_prob_all[0, self.true_label] / self._cnt_all),
            # Fraction of timepoints which correctly predicted given class
            'acc_all': float(self._acc_all[0, self.true_label] / self._cnt_all),
            # Simply take a majority vote and if it is correct assign 1
            'X_acc_all': 1.0 if np.argmax(self._acc_all, axis=1) == self.true_label else 0.0,
            # Take class with highest sum of log probabilities and if it is correct assign 1
            'X_acc_all_log_prob': 1.0 if np.argmax(self._log_prob_all, axis=1) == self.true_label else 0.0,

            # Average log likelihood loss for the correct class using only chunks end timepoints
            'log_loss_end': float(-self._log_prob_end[0, self.true_label] / self._cnt_end),
            # Fraction of data chunk end timepoints which correctly predicted given class
            'acc_end': float(self._acc_end[0, self.true_label] / self._cnt_end),
            # Simply take a majority vote and if it is correct assign 1
            'X_acc_end': 1.0 if np.argmax(self._acc_end, axis=1) == self.true_label else 0.0,
            # Take class with highest sum of log probabilities and if it is correct assign 1
            'X_acc_end_log_prob': 1.0 if np.argmax(self._log_prob_end, axis=1) == self.true_label else 0.0,

            # Log loss of the last prediction
            'log_loss_end_last': float(-self._log_prob_end_last[0, self.true_label]),
            # Accuracy using only last seen prediction from the last data chunk
            'X_acc_end_last': 1. if np.argmax(self._log_prob_end_last, axis=1) == self.true_label else 0.0
        }

        return res

    @staticmethod
    def average_stats(stats):
        """
        Takes statistics over multiple examples and creates one single aggregated description.
        """
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
            res[key] = sum(s[key] for s in stats) / example_cnt

        return res


class SingleLabelMetrics(BaseMetrics):
    """
    For normal/abnormal EEG dataset we have very long recordings and we might want to average predictions
    from the full recording instead of taking just the last one.
    """
    def __init__(self, name, output_size):
        super().__init__(Example, name, output_size)
