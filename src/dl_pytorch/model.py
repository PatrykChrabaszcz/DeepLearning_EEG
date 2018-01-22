import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from src.dl_pytorch.utils import BNGRU, BNLSTM
from src.dl_core.model import ModelBase
import logging


logger = logging.getLogger(__name__)


class PytorchModelBase(nn.Module, ModelBase):
    def __init__(self):
        super().__init__()

    @staticmethod
    def add_arguments(parser):
        return

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))

    def count_params(self):
        pp = 0
        for p in list(self.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            pp += nn
        return pp


class RnnBase(PytorchModelBase):
    cell_mapper = {
        'LSTM': nn.LSTM,
        'GRU': nn.GRU,
        'BNGRU': BNGRU,
        'BNLSTM': BNLSTM,
    }

    @staticmethod
    def add_arguments(parser):
        PytorchModelBase.add_arguments(parser)
        parser.add_argument("--hidden_size", dest="hidden_size", type=int, default=128,
                            help="Number of neurons in the RNN layer.",)
        parser.add_argument("--num_layers", dest="num_layers", type=int, default=3,
                            help="Number of layers in the RNN network.")
        parser.add_argument("--dropout", dest="dropout", type=float, default=0.0,
                            help="Dropout value.")
        parser.add_argument("--cell_type", dest="cell_type", type=str, choices=RnnBase.cell_mapper.keys(),
                            default='GRU',
                            help="RNN cell type.")
        parser.add_argument("--use_context", dest="use_context", type=int, choices=[0, 1], default=0,
                            help="If 1 then context information will be used.")

    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout, cell_type, use_context, context_size):
        super().__init__()
        args = dir(self)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout
        self.cell_type = cell_type
        self.use_context = use_context
        self.context_size = context_size

        args = [a for a in dir(self) if a not in args]
        for arg in args:
            logger.debug('RnnBase | attr: %s | val: %s' % (arg, getattr(self, arg)))

    def initial_state(self):
        if self.cell_type in ["LSTM", "BNLSTM"]:
            return np.array(np.random.normal(0, 1.0, (self.num_layers, self.hidden_size)), dtype=np.float32), \
                   np.array(np.random.normal(0, 1.0, (self.num_layers, self.hidden_size)), dtype=np.float32)
        elif self.cell_type in ["GRU", "BNGRU"]:
            #return np.zeros((self.num_layers, self.hidden_size), dtype=np.float32)
            random_state = np.random.normal(0, 1.0, (self.num_layers, self.hidden_size))
            return np.clip(random_state, -1, 1).astype(dtype=np.float32)
        else:
            raise NotImplementedError("Function initial_state() not implemented for cell type %s" % self.cell_type)

    # Converts PyTorch hidden state representation into something that can be saved
    def export_state(self, states):
        if self.cell_type in ["LSTM", "BNLSTM"]:
            states_0 = np.swapaxes(states[0].cpu().data.numpy(), 0, 1)
            states_1 = np.swapaxes(states[1].cpu().data.numpy(), 0, 1)

            assert (states_0.shape == states_1.shape)

            return [(a, b) for (a, b) in zip(states_0, states_1)]
        elif self.cell_type in ["GRU", "BNGRU"]:
            states = np.swapaxes(states.cpu().data.numpy(), 0, 1)
            return [s for s in states]
        else:
            raise NotImplementedError

    # Converts PyTorch hidden state representation into something that can be saved
    def import_state(self, states):
        if self.cell_type in ["LSTM", "BNLSTM"]:
            states_0, states_1 = np.stack([s[0] for s in states]), np.stack([s[1] for s in states])
            states_0, states_1 = np.swapaxes(states_0, 1, 0), np.swapaxes(states_1, 1, 0)

            states_0, states_1 = Variable(torch.from_numpy(states_0), requires_grad=False),\
                                 Variable(torch.from_numpy(states_1), requires_grad=False)

            return states_0, states_1

        elif self.cell_type in ["GRU", "BNGRU"]:
            states = np.stack(states)
            states = np.swapaxes(states, 1, 0)
            states = Variable(torch.from_numpy(states), requires_grad=False)
            return states
        else:
            raise NotImplementedError


class SimpleRNN(RnnBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        input_size = self.input_size if not self.use_context else self.input_size + self.context_size
        NetworkClass = self.cell_mapper[self.cell_type]
        self.rnn = NetworkClass(input_size=input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                                batch_first=True, dropout=self.dropout)
        self.dropout_layer = nn.Dropout(p=self.dropout)

        self.fc = nn.Linear(in_features=self.hidden_size, out_features=self.output_size)


    def forward(self, x, hidden, context):
        if self.use_context:
            context = torch.cat([context] * x.size()[1], dim=1)
            x = torch.cat([x, context], dim=2)

        lstm_out, hidden = self.rnn(x, hidden)
        lstm_out = lstm_out.contiguous()
        lstm_out = self.dropout_layer(lstm_out)

        fc_out = self.fc(lstm_out.view(lstm_out.size(0) * lstm_out.size(1), lstm_out.size(2)))
        fc_out = fc_out.view(lstm_out.size(0), lstm_out.size(1), fc_out.size(1))
        return fc_out, hidden

    def offset_size(self):
        return 0


class ConvRNN(RnnBase):
    def __init__(self, cnn_size=10, **kwargs):
        super().__init__(**kwargs)

        k_s = 10

        self.offset = sum([cnn_kernel_size - 1 for cnn_kernel_size in cnn_kernel_sizes])

        NetworkClass = self.cell_mapper[self.cell_type]

        # For each input channel
        self.rnn = NetworkClass(input_size=self.input_size*k_s, hidden_size=self.hidden_size, num_layers=self.num_layers,
                                batch_first=True, dropout=self.dropout)
        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.fc = nn.Linear(in_features=self.hidden_size, out_features=self.output_size)

        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=cnn_size, kernel_size=(k_s, 1))
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=cnn_size, kernel_size=(k_s, 1))
        self.conv_1_dropout = nn.Dropout(self.dropout)
        self.non_lin_1 = nn.ReLU()

    def forward(self, x, hidden, context):
        # x Has shape [batch_size, sequence_length, input_dim]
        x = x.unsqueeze(1)
        # Now x Has shape [batch_size, 1, sequence_length, input_dim]
        x = self.conv_1(x)
        x = self.conv_1_dropout(x)
        x = self.non_lin_1(x)
        # Now x Has shape [batch_size, k_s, sequence_length, input_dim]
        x = x.transpose(1, 2).contiguous()
        # Now x Has shape [batch_size, sequence_length, k_s,  input_dim]
        x = x.view(x.size(0), x.size(1), x.size(2)*x.size(3))
        # Now x Has shape [batch_size, sequence_length, k_s * input_dim]

        lstm_out, hidden = self.rnn(x, hidden)
        lstm_out = self.dropout_layer(lstm_out)
        lstm_out = lstm_out.contiguous()
        fc_out = self.fc(lstm_out.view(lstm_out.size(0) * lstm_out.size(1), lstm_out.size(2)))
        fc_out = fc_out.view(lstm_out.size(0), lstm_out.size(1), fc_out.size(1))
        print(fc_out.size())
        return fc_out, hidden

    def offset_size(self):
        return self.offset




# class BNLSTM(SimpleRNN):
#     def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout, cell_type):
#         super().__init__(input_size, hidden_size, num_layers, num_classes, dropout, cell_type)
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.cell_type = cell_type
#
#         if self.cell_type == 'BNLSTM':
#             self.lstm = utils.LSTM(cell_class=utils.BNLSTMCell, input_size=22,
#                                hidden_size=hidden_size, batch_first=True)
#         elif self.cell_type == 'LSTM':
#             self.lstm = utils.LSTM(cell_class=utils.LSTMCell, input_size=22,
#                          hidden_size=hidden_size, batch_first=True)
#
#         self.fc = nn.Linear(hidden_size, num_classes)
#
#     def forward(self, x, hidden):
#         lstm_out, hidden = self.lstm(x, hidden)
#         lstm_out = lstm_out.contiguous()
#         fc_out = self.fc(lstm_out.view(lstm_out.size(0) * lstm_out.size(1), lstm_out.size(2)))
#         fc_out = fc_out.view(lstm_out.size(0), lstm_out.size(1), fc_out.size(1))
#         return fc_out, hidden