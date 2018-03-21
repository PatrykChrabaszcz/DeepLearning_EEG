from src.result_processing.base_metrics import BaseMetrics
import numpy as np


class Example:
    def __init__(self, example_id, output_size):
        self.example_id = example_id
        self.output_size = output_size

        self._cnt_all = 0
        self._cnt_end = 0

        self._l2_loss_all = np.zeros([1, output_size])
        self._l1_loss_all = np.zeros([1, output_size])

        self._l2_loss_end = np.zeros([1, output_size])
        self._l1_loss_end = np.zeros([1, output_size])

        self._l2_loss_end_last = np.zeros([1, output_size])
        self._l1_loss_end_last = np.zeros([1, output_size])

        self._last_prediction = np.zeros([1, output_size])
        self._mean_prediction = np.zeros([1, output_size])

        self._running_covariance = RunningCovariance(output_size)
        self._static_covariance = RunningCovariance(output_size)

    # Output should have shape [timepoints x labels_cnt]
    def append(self, output, labels):
        assert len(output) == len(labels), 'Size mismatch for outputs and labels'
        assert output.shape[1] == self.output_size, \
            'Number of labels %d not consistent with the network output %d' % (self.output_size, output.shape[0])

        # Number of timepoints in this data chunk
        num_timepoints = len(output)

        # Update loss based on all timepoints
        self._l2_loss_all += np.sum(np.square(output-labels), axis=0)
        self._l1_loss_all += np.sum(np.abs(output-labels), axis=0)

        # Update loss based on the last timepoints from data chunks
        self._l2_loss_end += np.square(output[[-1], :] - labels[[-1], :])
        self._l1_loss_end += np.abs(output[[-1], :] - labels[[-1], :])

        # Also save loss of the last timepoint from the last data chunk
        self._l2_loss_end_last = np.square(output[[-1], :] - labels[[-1], :])
        self._l1_loss_end_last = np.abs(output[[-1], :] - labels[[-1], :])

        # Mean prediction will be later divided by cnt all to get the average prediction over all timepoints
        # Only makes sense to look at this if whole recording has the same label, for example Age for EEG Abnormal
        self._mean_prediction += np.sum(output, axis=0)

        # Similar to the mean prediction but report only the last prediction from the RNN instead of the
        self._last_prediction = output[[-1], :]

        # Number of timepoints seen for this example
        self._cnt_all += num_timepoints
        # Number of data chunks seen for this example
        self._cnt_end += 1

        self._running_covariance.append(output, labels)
        self._static_covariance.append(output, labels)

    def stats(self):
        res = {
            'cnt_all': self._cnt_all,
            'cnt_end': self._cnt_end,

            'l1_loss_all': (self._l1_loss_all / self._cnt_all).flatten().tolist(),
            'l2_loss_all': (self._l2_loss_all / self._cnt_all).flatten().tolist(),

            'l1_loss_end': (self._l1_loss_end / self._cnt_end).flatten().tolist(),
            'l2_loss_end': (self._l2_loss_end / self._cnt_end).flatten().tolist(),

            'l1_loss_end_last': self._l1_loss_end_last.flatten().tolist(),
            'l2_loss_end_last': self._l2_loss_end_last.flatten().tolist(),

            'mean_prediction': (self._mean_prediction / self._cnt_all).flatten().tolist(),
            'last_prediction': self._last_prediction.flatten().tolist(),

            'correlation': self._running_covariance.corr().flatten().tolist(),
            'correlation_s': self._static_covariance.corr().flatten().tolist()
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
            try:
                res[key] = sum([s[key] for s in stats]) / example_cnt
            except TypeError:
                res[key] = [sum(b)/example_cnt for b in zip(*[s[key] for s in stats])]
        return res


class RunningCovariance:
    """
    This class is used to compute running correlation statistics between predictions and targets. For each
    (prediction, target) pair we keep the track of required statistics and used them to compute
    corresponding correlation when requested.
    """
    def __init__(self, dim):
        self.x_sum = np.zeros(dim)
        self.y_sum = np.zeros(dim)

        self.xx = np.zeros(dim)
        self.yy = np.zeros(dim)
        self.xy = np.zeros(dim)
        self.cnt = 0

    def append(self, x, y):
        assert x.shape[0] == y.shape[0]
        assert x.shape[1] == y.shape[1]

        n = x.shape[0]
        x_s = np.sum(x, axis=0)
        y_s = np.sum(y, axis=0)

        self.x_sum = self.x_sum + x_s
        self.y_sum = self.y_sum + y_s

        self.xx += np.sum(x**2, axis=0)
        self.yy += np.sum(y**2, axis=0)
        self.xy += np.sum(x*y, axis=0)

        self.cnt += n

    def corr(self):
        return self.xy/self.cnt - self.x_sum*self.y_sum/(self.cnt**2)

    def x_var(self):
        return self.xx/self.cnt - (self.x_sum/self.cnt)**2

    def y_var(self):
        return self.yy/self.cnt - (self.y_sum/self.cnt)**2


class StaticCovariance:
    """
    This class is used to compute running correlation statistics between predictions and targets. For each
    (prediction, target) pair we keep the track of required statistics and used them to compute
    corresponding correlation when requested.
    """
    def __init__(self, dim):
        self.dim = dim
        self.x = []
        self.y = []

    def append(self, x, y):
        self.x.append(x)
        self.y.append(y)

    def corr(self):
        x = np.concatenate(self.x)
        y = np.concatenate(self.y)

        c = []
        for i in range(self.dim):
            c.append(np.cov(x[:, i], y[:, i]))

        return np.array([e[0][1] / np.sqrt(e[0][0]*e[1][1]) for e in c])

    def x_var(self):
        raise NotImplementedError('Not now')

    def y_var(self):
        raise NotImplementedError('Not implemented yet')


class RegressionMetrics(BaseMetrics):
    def __init__(self, name, output_size):
        super().__init__(Example, name, output_size)


# Would be more professional to write unit tests.
# Right now only some simple user assisted sanity checks are provided.
if __name__ == '__main__':

    def test_running_covariance():
        np.random.seed(0)
        x = np.random.normal(size=[100000, 5])
        y = np.random.normal(size=[100000, 5])

        c = []
        for i in range(5):
            c.append(np.cov(x[:, i], y[:, i]))

        rc = RunningCovariance(dim=5)
        rc_online = RunningCovariance(dim=5)
        rc.append(x, y)
        rc_online.append(x[:x.shape[0]//2, :], y[:y.shape[0]//2, :])
        rc_online.append(x[x.shape[0]//2:, :], y[y.shape[0]//2:, :])

        x_var = [e[0][0] for e in c]

        print('\nX_Var')
        print('Numpy:')
        print(x_var)
        print('Running Convariance Class:')
        print(rc.x_var())
        print('Running Convariance Class Online:')
        print(rc_online.x_var())

        xy_covar = [e[0][1] / np.sqrt(e[0][0]*e[1][1]) for e in c]

        print('\nXY_Corr')
        print('Numpy:')
        print(xy_covar)
        print('Running Convariance Class:')
        print(rc.corr())
        print('Running Convariance Class Online:')
        print(rc_online.corr())

    def test_regression_metrics():
        metrics = RegressionMetrics('Metrics', 2)

        example_ids = [1, 2, 3]
        output = np.array([
            [[1, 1], [1, 1], [1, 1], [1, 2]],
            [[2, 2], [2, 2], [2, 2], [2, 2]],
            [[3, 2], [3, 3], [3, 3], [3, 2]]
        ])
        labels = np.array([
            [[1, 1], [1, 1], [1, 1], [1, 1]],
            [[2, 2], [2, 2], [2, 2], [2, 2]],
            [[3, 3], [3, 3], [3, 3], [3, 3]]
        ])
        metrics.append_results(example_ids, output, labels, 0, 3)

        results = metrics.get_summarized_results()
        print(results)


    test_running_covariance()







