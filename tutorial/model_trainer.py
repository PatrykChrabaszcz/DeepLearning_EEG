from src.dl_core import ModelTrainerBase
from src.dl_core import RegressionMetrics
from torch.autograd import Variable
from torch import nn
import torch


class ModelTrainer(ModelTrainerBase):
    def __init__(self, model, learning_rate, weight_decay,
                 train_dr, test_dr, sequence_size, loss_type):
        super().__init__(model, learning_rate, weight_decay, train_dr, test_dr, sequence_size, loss_type)

        self.cuda = False

        if self.cuda:
            self.model = self.model.cuda()

        self.metrics = RegressionMetrics()
        self.criterion = nn.MSELoss()

        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=self.weight_decay)

    def _one_iteration(self, batch, time, hidden, labels, update=False):
        # Forward pass
        batch = Variable(torch.from_numpy(batch))
        labels = Variable(torch.from_numpy(labels))
        if self.cuda:
            batch = batch.cuda()
            hidden = hidden[0].cuda(), hidden[1].cuda()
            labels = labels.cuda()

        outputs, hidden = self.model(batch, hidden)

        # Backward pass
        if update:
            if self.loss_type == 'classification_last':
                training_outputs = outputs[:, -1, :]
                training_labels = labels[:, -1]
            elif self.loss_type == 'classification_all':
                outputs_num = outputs.size()[-1]
                training_outputs = outputs.view(-1, outputs_num)
                training_labels = labels.view(-1)
            else:
                raise NotImplementedError

            loss = self.criterion(training_outputs, training_labels)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(self.model.parameters(), 0.25)
            self.optimizer.step()

        return outputs, hidden

    def _gather_results(self, ids, outputs, labels, train=True):
        self.metrics.append_results(ids, outputs.cpu().data.numpy(), labels, train=train)

