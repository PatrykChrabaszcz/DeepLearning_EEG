from src.dl_core.model_trainer import ModelTrainerBase
from src.dl_core.model import ModelBase
from torch.autograd import Variable
from torch import nn
import torch


class ModelTrainer(ModelTrainerBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cuda = True
        if 'CrossEntropy' in self.objective_type:
            self.criterion = nn.CrossEntropyLoss()
        elif 'MeanSquaredError' in self.objective_type:
            self.criterion = nn.MSELoss()
        elif 'L1Loss' in self.objective_type:
            self.criterion = nn.L1Loss()
        else:
            raise NotImplementedError('This objective type is not implemented inside the ModelTrainer class')

        if self.cuda:
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()

        if self.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate,
                                              weight_decay=self.weight_decay)
        elif self.optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9,
                                             weight_decay=self.weight_decay)
        else:
            raise NotImplementedError('This optimizer is not implemented')

    def _one_iteration(self, batch, time, hidden, labels, context, update=False):
        if update:
            self.model.train()
        else:
            self.model.eval()

        # Forward pass
        batch = Variable(torch.from_numpy(batch))
        labels = Variable(torch.from_numpy(labels))

        if None not in context:
            context = Variable(torch.from_numpy(context))

        if self.cuda:
            batch = batch.cuda()
            labels = labels.cuda()
            if isinstance(hidden, tuple):
                hidden = (h.cuda() for h in hidden)
            else:
                hidden = hidden.cuda()
            if self.model.context_size > 0:
                context = context.cuda()

        outputs, hidden = self.model(batch, hidden, context)

        if '_last' in self.objective_type:
            training_outputs = outputs[:, -1, :]
            training_labels = labels[:, -1]
        elif '_all' in self.objective_type:
            outputs_num = outputs.size()[-1]
            training_outputs = outputs.view(-1, outputs_num)
            training_labels = labels.view(-1)
        else:
            raise NotImplementedError

        loss = self.criterion(training_outputs, training_labels)
        # Backward pass
        if update:
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(self.model.parameters(), 1)

            self.optimizer.step()

        #     total_norm = 0
        #     for p in self.model.parameters():
        #         param_norm = p.grad.data.norm(2)
        #         total_norm += param_norm ** 2
        #     total_norm = total_norm ** (1. / 2)
        #     print('Gradient norm: ', total_norm)

        return outputs, hidden, loss

    def _gather_results(self, ids, outputs, labels, loss, batch_size, metrics):
        metrics.append_results(ids, outputs.cpu().data.numpy(), labels, loss.cpu().data.numpy()[0],
                               batch_size=batch_size)

