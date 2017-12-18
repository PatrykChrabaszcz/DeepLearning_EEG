from torch.autograd import Variable
from torch import nn
import torch
from src.core.metrics import Metrics


class ModelTrainer:
    def __init__(self, model, train_dr, test_dr, sequence_size):
        self.model = model.cuda()
        self.train_dr = train_dr
        self.test_dr = test_dr
        self.metrics = Metrics()

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

        self.sequence_size = sequence_size

        self.train_dr.initialize_epoch(randomize=True, sequence_size=self.sequence_size,
                                       initial_state=self.model.initial_hidden())
        self.test_dr.initialize_epoch(randomize=False, sequence_size=self.sequence_size,
                                      initial_state=self.model.initial_hidden())

        self.train_dr.start_readers()
        self.test_dr.start_readers()

    def process_one_epoch(self, train=True):
        dr = self.train_dr if train else self.test_dr
        total_loss = 0
        iteration = 0
        try:
            while True:
                batch, labels, ids = dr.get_batch()
                batch = Variable(torch.from_numpy(batch)).cuda()
                labels = Variable(torch.from_numpy(labels)).cuda()
                hidden = self.model.import_hidden(dr.get_states(ids), cuda=True)
                output, hidden = self.model(batch, hidden)

                loss = self.criterion(output, labels)
                total_loss += loss.cpu().data.numpy()[0]

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm(self.model.parameters(), 0.001)
                    self.optimizer.step()

                self.metrics.append_results(ids, output.cpu().data.numpy(), labels.cpu().data.numpy(), train=train)

                dr.set_states(ids, self.model.export_hidden(hidden))

                iteration += 1
                if iteration % 100 is 0:
                    print('Iterations done %d' % iteration)

        except IndexError:
            print('%d Iterations in this epoch' % iteration)

            if iteration > 0:
                print('Loss %g' % (total_loss / iteration))

            self.metrics.finish_epoch(train=train)

            dr.initialize_epoch(randomize=train, sequence_size=self.sequence_size,
                                initial_state=self.model.initial_hidden())

