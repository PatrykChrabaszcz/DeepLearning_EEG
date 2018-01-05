from src.data_reader import SequenceDataReader
from torch.autograd import Variable
from torch import nn
import torch
from src.core.metrics import ClassificationMetrics
from src.utils import Stats


class ModelTrainer:
    LAST_OUTPUT = 0
    ALL_OUTPUTS = 0

    def __init__(self, model, learning_rate, train_dr, test_dr, sequence_size, forget_state, loss_type=LAST_OUTPUT):
        self.model = model.cuda()
        self.train_dr = train_dr
        self.test_dr = test_dr
        self.forget_state = forget_state
        self.loss_type = loss_type
        self.metrics = ClassificationMetrics()

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        self.sequence_size = sequence_size

        self.train_dr.initialize_epoch(randomize=True, sequence_size=self.sequence_size)
        self.test_dr.initialize_epoch(randomize=False, sequence_size=self.sequence_size)

        self.train_dr.start_readers()
        self.test_dr.start_readers()

    def process_one_epoch(self, train=True):
        dr = self.train_dr if train else self.test_dr
        iteration = 0
        time_stats = Stats('Time Statistics')
        get_batch_stats = time_stats.create_child_stats('Get Batch')
        forward_pass_stats = time_stats.create_child_stats('Forward Pass')
        backward_pass_stats = time_stats.create_child_stats('Backward Pass')
        process_metrics_stats = time_stats.create_child_stats('Process Metrics')
        save_states_stats = time_stats.create_child_stats('Save States')
        try:
            with time_stats:
                while True:
                    with get_batch_stats:
                        batch, time, labels, ids = dr.get_batch()
                        batch = Variable(torch.from_numpy(batch)).cuda()
                        labels = Variable(torch.from_numpy(labels)).cuda()
                        hidden = self.model.import_state(dr.get_states(ids, forget=self.forget_state), cuda=True)

                    with forward_pass_stats:
                        outputs, hidden = self.model(batch, hidden)

                    with backward_pass_stats:
                        if train:
                            if self.loss_type == self.LAST_OUTPUT:
                                # Take the last prediction for loss estimate
                                last_output = outputs[:, -1, :]
                                # Take the last label for loss estimate
                                last_label = labels[:, -1]
                                loss = self.criterion(last_output, last_label)
                            elif self.loss_type == self.ALL_OUTPUTS:
                                loss = self.criterion(outputs, labels)

                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()

                    with process_metrics_stats:
                        self.metrics.append_results(ids, outputs.cpu().data.numpy(), labels.cpu().data.numpy(), train=train)

                    with save_states_stats:
                        dr.set_states(ids, self.model.export_state(hidden))

                    iteration += 1
                    if iteration % 100 is 0:
                        print('Iterations done %d' % iteration)

        except SequenceDataReader.EpochDone:
            print('%d Iterations in this epoch' % iteration)

            result = self.metrics.finish_epoch(train=train)

            dr.initialize_epoch(randomize=train, sequence_size=self.sequence_size)

            return result
