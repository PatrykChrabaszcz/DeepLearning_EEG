import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from src.dl_core.model import ModelBase
from torch import optim
import numpy as np
import math, random


class TutorialRNN(nn.Module):
    cell_mapper = {
        'LSTM': nn.LSTM
    }

    def __init__(self, input_size, hidden_size, num_layers, num_classes, cell_type):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = self.cell_mapper[cell_type](input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, hidden):
        lstm_out, hidden = self.lstm(x, hidden)
        lstm_out = lstm_out.contiguous()
        fc_out = self.fc(lstm_out.view(lstm_out.size(0) * lstm_out.size(1), lstm_out.size(2)))
        fc_out = fc_out.view(lstm_out.size(0), lstm_out.size(1), fc_out.size(1))
        return fc_out, hidden

    def initial_state(self):
        return np.array(np.random.normal(0, 1.0, (self.num_layers, self.hidden_size)), dtype=np.float32), \
               np.array(np.random.normal(0, 1.0, (self.num_layers, self.hidden_size)), dtype=np.float32)

    # Converts PyTorch hidden state representation into something that can be saved
    def export_state(self, states):
        states_0 = np.swapaxes(states[0].cpu().data.numpy(), 0, 1)
        states_1 = np.swapaxes(states[1].cpu().data.numpy(), 0, 1)

        assert (states_0.shape == states_1.shape)

        return [(a, b) for (a, b) in zip(states_0, states_1)]

    # Converts PyTorch hidden state representation into something that can be saved
    def import_state(self, states):
        states_0, states_1 = np.stack([s[0] for s in states]), np.stack([s[1] for s in states])
        states_0, states_1 = np.swapaxes(states_0, 1, 0), np.swapaxes(states_1, 1, 0)

        states_0, states_1 = Variable(torch.from_numpy(states_0), requires_grad=False),\
                             Variable(torch.from_numpy(states_1), requires_grad=False)

        return states_0, states_1

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))

    def count_params(self):
        pp = 0
        for p in list(self.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn*s
            pp += nn
        return pp
