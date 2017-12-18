import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch import optim
import numpy as np
import math, random


class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3, num_classes=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, hidden):
        # Forward propagate RNN
        out, hidden = self.lstm(x, hidden)
        # Decode hidden state of last time step
        out = self.fc(out[:, -1, :])
        return out, hidden

    def initial_hidden(self):
        return np.array(np.random.normal(0, 1.0, (self.num_layers, self.hidden_size)), dtype=np.float32), \
               np.array(np.random.normal(0, 1.0, (self.num_layers, self.hidden_size)), dtype=np.float32)

    # Converts PyTorch hidden state representation into something that can be saved
    def export_hidden(self, states):
        states_0 = np.swapaxes(states[0].cpu().data.numpy(), 0, 1)
        states_1 = np.swapaxes(states[1].cpu().data.numpy(), 0, 1)

        assert (states_0.shape == states_1.shape)

        return [(a, b) for (a, b) in zip(states_0, states_1)]

    # Converts PyTorch hidden state representation into something that can be saved
    def import_hidden(self, states, cuda=True):
        states_0, states_1 = np.stack([s[0] for s in states]), np.stack([s[1] for s in states])
        states_0, states_1 = np.swapaxes(states_0, 1, 0), np.swapaxes(states_1, 1, 0)

        states_0, states_1 = Variable(torch.from_numpy(states_0), requires_grad=False),\
                             Variable(torch.from_numpy(states_1), requires_grad=False)
        if cuda:
            return states_0.cuda(), states_1.cuda()
        else:
            return states_0, states_1

    def count_params(self):
        pp = 0
        for p in list(self.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn*s
            pp += nn
        return pp


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.sequence_length = 500

        self.conv1 = nn.Conv2d(1, 1, (10, 1), stride=(2, 1))
        self.conv2 = nn.Conv2d(1, 1, (10, 1), stride=(2, 1))
        self.conv3 = nn.Conv2d(1, 64, (4, 22), stride=(2, 1))
        self.conv4 = nn.Conv2d(1, 128, (4, 22), stride=(2, 1))

        #self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, hidden):
        print('Forward')
        # X is assumed to have format [Batch Size, Time, Channels]
        # Need to reshape to [Batch Size, 1, Time, Channels]

        timesteps = x.size()[1]
        channels = x.size()[2]
        x = x.view(-1, 1, timesteps, channels)

        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)

        x = x.view(x.size(0), -1)

        print(x)
        return 1

    def initial_hidden(self):
        return np.zeros([1, 1]), np.zeros([1, 1])
