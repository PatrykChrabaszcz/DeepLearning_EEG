from torch.autograd import Variable
from torch import nn
import torch
from src.core.metrics import ClassificationMetrics, RegressionMetrics
from src.data_reader import SequenceDataReader


class ModelTrainer:
    def __init__(self, model, learning_rate, train_dr, test_dr, sequence_size, forget_state):
        self.model = model
        self.train_dr = train_dr
        self.test_dr = test_dr
        self.forget_state = forget_state
        self.metrics = RegressionMetrics()

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        self.sequence_size = sequence_size

        self.train_dr.initialize_epoch(randomize=True, sequence_size=self.sequence_size)
        self.test_dr.initialize_epoch(randomize=False, sequence_size=self.sequence_size)

        self.train_dr.start_readers()
        self.test_dr.start_readers()

    def process_one_epoch(self, train=True):
        dr = self.train_dr if train else self.test_dr
        iteration = 0
        try:
            while True:
                batch, time, labels, ids = dr.get_batch()
                batch = Variable(torch.from_numpy(batch))
                labels = Variable(torch.from_numpy(labels))
                hidden = self.model.import_state(dr.get_states(ids, forget=self.forget_state))
                outputs, hidden = self.model(batch, hidden)

                # Take the last prediction for loss estimate
                last_output = outputs[:, -1, :]
                # Take the last label for loss estimate
                last_label = labels[:, -1]

                loss = self.criterion(last_output, last_label)

                #loss = self.criterion(outputs, labels)
                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                self.metrics.append_results(ids, outputs.data.numpy(), labels.data.numpy(), train=train)

                dr.set_states(ids, self.model.export_state(hidden))

                iteration += 1
                if iteration % 100 is 0:
                    print('Iterations done %d' % iteration)

        except SequenceDataReader.EpochDone:
            print('%d Iterations in this epoch' % iteration)

            result = self.metrics.finish_epoch(train=train)

            dr.initialize_epoch(randomize=train, sequence_size=self.sequence_size)

            return result
