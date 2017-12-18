
class Metrics:
    def __init__(self):
        self.train_results = {}
        self.test_results = {}

    def append_results(self, ids, output, labels, train=True):
        results = self.train_results if train else self.test_results

        for i, o, l in zip(ids, output, labels):
            if i in results:
                results[i].append((o, l))
            else:
                results[i] = [(o, l)]

    def finish_epoch(self, train=True):
        results = self.train_results if train else self.test_results

        good_last = 0
        bad_last = 0

        good_average = 0
        bad_average = 0
        for i, l in results.items():
            predictions = [r[0].argmax() for r in l]
            labels = [r[1] for r in l]

            v = [1 if p == lab else 0 for (p, lab) in zip(predictions, labels)]

            if predictions[-1] == labels[-1]:
                good_last += 1
            else:
                bad_last += 1

            if sum(v) > len(v)/2:
                good_average += 1
            else:
                bad_average += 1

        print('Results for %s' % ('Train' if train else 'Test'))
        print('Good %d \t, Bad %d' % (good_last, bad_last))
        print('GoodAv %d \t, BadAv %d' % (good_average, bad_average))

        results.clear()
