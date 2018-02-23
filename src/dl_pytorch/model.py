import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from src.dl_core.model import ModelBase
import logging
from src.dl_pytorch.utils import SeparateChannelCNN, AllChannelsCNN, RNN
from torchviz import make_dot


logger = logging.getLogger(__name__)


class PytorchModelBase(nn.Module, ModelBase):
    Skip_None = 'none'
    Skip_Add = 'add'
    Skip_Concat = 'concat'

    def __init__(self, batch_norm, skip_mode, **kwargs):
        super().__init__()
        self.batch_norm = batch_norm
        self.skip_mode = skip_mode

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--batch_norm", dest="batch_norm", type=int, default=0, choices=[0, 1],
                            help="Whether to use batch norm or not", )
        parser.add_argument("--skip_mode", dest="skip_mode", type=str, default='none',
                            choices=['none', 'add', 'concat'],
                            help="Whether to skip connections", )
        return parser

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
    }

    @staticmethod
    def add_arguments(parser):
        PytorchModelBase.add_arguments(parser)
        parser.add_argument("--rnn_hidden_size", dest="rnn_hidden_size", type=int, default=128,
                            help="Number of neurons in the RNN layer.")
        parser.add_argument("--rnn_num_layers", dest="rnn_num_layers", type=int, default=3,
                            help="Number of layers in the RNN network.")
        parser.add_argument("--dropout", dest="dropout", type=float, default=0.0,
                            help="Dropout value.")
        parser.add_argument("--rnn_cell_type", dest="rnn_cell_type", type=str, choices=RnnBase.cell_mapper.keys(),
                            default='GRU',
                            help="RNN cell type.")
        parser.add_argument("--use_context", dest="use_context", type=int, choices=[0, 1], default=0,
                            help="If 1 then context information will be used.")
        return parser

    def __init__(self, input_size, rnn_hidden_size, rnn_num_layers, output_size, dropout,
                 rnn_cell_type, use_context, context_size, **kwargs):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layers = rnn_num_layers
        self.output_size = output_size
        self.dropout = dropout
        self.rnn_cell_type = rnn_cell_type
        self.use_context = use_context
        self.context_size = context_size

    def initial_state(self):
        if self.rnn_cell_type == "LSTM":
            return np.array(np.random.normal(0, 1.0, (self.rnn_num_layers, self.rnn_hidden_size)), dtype=np.float32), \
                   np.array(np.random.normal(0, 1.0, (self.rnn_num_layers, self.rnn_hidden_size)), dtype=np.float32)
        elif self.rnn_cell_type == "GRU":
            random_state = np.random.normal(0, 1.0, (self.rnn_num_layers, self.rnn_hidden_size))
            return np.clip(random_state, -1, 1).astype(dtype=np.float32)
        else:
            raise NotImplementedError("Function initial_state() not implemented for cell type %s" % self.rnn_cell_type)

    # Converts PyTorch hidden state representation into something that can be saved
    def export_state(self, states):
        if self.rnn_cell_type == "LSTM":
            states_0 = np.swapaxes(states[0].cpu().data.numpy(), 0, 1)
            states_1 = np.swapaxes(states[1].cpu().data.numpy(), 0, 1)

            assert (states_0.shape == states_1.shape)

            return [(a, b) for (a, b) in zip(states_0, states_1)]
        elif self.rnn_cell_type == "GRU":
            states = np.swapaxes(states.cpu().data.numpy(), 0, 1)
            return [s for s in states]
        else:
            raise NotImplementedError

    # Convert something that was saved into PyTorch representation
    def import_state(self, states):
        if self.rnn_cell_type == "LSTM":
            states_0, states_1 = np.stack([s[0] for s in states]), np.stack([s[1] for s in states])
            states_0, states_1 = np.swapaxes(states_0, 1, 0), np.swapaxes(states_1, 1, 0)

            states_0, states_1 = Variable(torch.from_numpy(states_0), requires_grad=False),\
                                 Variable(torch.from_numpy(states_1), requires_grad=False)
            return states_0, states_1

        elif self.rnn_cell_type == "GRU":
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

        cell = RnnBase.cell_mapper[self.rnn_cell_type]
        self.rnn = RNN(cell=cell, in_size=input_size, hidden_size=self.rnn_hidden_size, num_layers=self.rnn_num_layers,
                       dropout=self.dropout, batch_norm=self.batch_norm, skip_mode=self.skip_mode)

        # self.dropout_layer = nn.Dropout(p=self.dropout)

        self.fc = nn.Linear(in_features=self.rnn_hidden_size, out_features=self.output_size, bias=True)

    def forward(self, x, hidden, context):
        if self.use_context:
            context = torch.cat([context] * x.size()[1], dim=1)
            x = torch.cat([x, context], dim=2)

        lstm_out, hidden = self.rnn(x, hidden)
        lstm_out = lstm_out.contiguous()
        #lstm_out = self.dropout_layer(lstm_out)

        fc_out = self.fc(lstm_out.view(lstm_out.size(0) * lstm_out.size(1), lstm_out.size(2)))
        fc_out = fc_out.view(lstm_out.size(0), lstm_out.size(1), fc_out.size(1))
        return fc_out, hidden

    def offset_size(self, sequence_size):
        return 0

    # def draw(self):
    #     x = Variable(torch.from_numpy(np.random.uniform(size=[64, 1, self.input_size]).astype(np.float32)))
    #     y, h = self(x, None, None)
    #     print('Make dot')
    #     dot = make_dot(y, params=dict(self.named_parameters()))
    #     print('Start render')
    #     dot.render('model.gv', view=False)
    #     print('Render done')


class ConvRNN(RnnBase):
    @staticmethod
    def add_arguments(parser):
        RnnBase.add_arguments(parser)
        parser.add_argument("--cnn_c_layers", dest="cnn_c_layers", type=int, default=3,
                            help="Number of layers in the first cnn block. "
                                 "Each channel processed separately the same filters.")
        parser.add_argument("--cnn_c_channels", dest="cnn_c_channels", type=int, default=10,
                            help="Number of filters in the first cnn block.")
        parser.add_argument("--cnn_c_width", dest="cnn_c_width", type=int, default=10,
                            help="Width in time dimension of the kernels in the first cnn block.")
        parser.add_argument("--cnn_c_stride", dest="cnn_c_stride", type=int, default=2,
                            help="Stride for the first CNN block.")

        parser.add_argument("--cnn_f_layers", dest="cnn_f_layers", type=int, default=3,
                            help="Number of layers in the second cnn block. "
                                 "All channels processed together")
        parser.add_argument("--cnn_f_channels", dest="cnn_f_channels", type=int, default=10,
                            help="Number of filters in the second cnn block.")
        parser.add_argument("--cnn_f_width", dest="cnn_f_width", type=int, default=10,
                            help="Width in time dimension of the kernels in the second cnn block.")
        parser.add_argument("--cnn_f_stride", dest="cnn_f_stride", type=int, default=2,
                            help="Stride for the second CNN block.")
        return parser

    def __init__(self, cnn_c_layers, cnn_c_channels, cnn_c_width, cnn_c_stride,
                 cnn_f_layers, cnn_f_channels, cnn_f_width, cnn_f_stride, **kwargs):

        super().__init__(**kwargs)

        out_size = self.input_size * cnn_c_channels
        self.cnn_c = SeparateChannelCNN(in_size=self.input_size, out_size=out_size, num_layers=cnn_c_layers,
                                        kernel_size=cnn_c_width, stride=cnn_c_stride, batch_norm=True)

        self.cnn_f = AllChannelsCNN(in_size=out_size, out_size=cnn_f_channels, num_layers=cnn_f_layers,
                                    kernel_size=cnn_f_width, stride=cnn_f_stride, batch_norm=True)

        cell = RnnBase.cell_mapper[self.rnn_cell_type]
        self.rnn = RNN(cell=cell, in_size=cnn_f_channels, hidden_size=self.rnn_hidden_size,
                       num_layers=self.rnn_num_layers, dropout=self.dropout, batch_norm=self.batch_norm)

        # self.dropout_layer = nn.Dropout(p=self.dropout)

        self.fc = nn.Linear(in_features=self.rnn_hidden_size, out_features=self.output_size, bias=True)

    def forward(self, x, hidden, context):
        x = self.cnn_c(x)
        x = self.cnn_f(x)

        lstm_out, hidden = self.rnn(x, hidden)
        lstm_out = lstm_out.contiguous()
        # lstm_out = self.dropout_layer(lstm_out)

        fc_out = self.fc(lstm_out.view(lstm_out.size(0) * lstm_out.size(1), lstm_out.size(2)))
        fc_out = fc_out.view(lstm_out.size(0), lstm_out.size(1), fc_out.size(1))
        return fc_out, hidden

    def offset_size(self, sequence_size):
        out_seq_size = self.cnn_c.out_seq_size(sequence_size)
        out_seq_size = self.cnn_f.out_seq_size(out_seq_size)
        return sequence_size - out_seq_size


class ChronoNet(RnnBase):
    class InceptionBlock(nn.Module):
        def __init__(self, in_size, out_size=32):
            super().__init__()
            self.conv_1 = nn.Conv1d(in_channels=in_size, out_channels=out_size, kernel_size=2, stride=2, padding=0)
            self.conv_2 = nn.Conv1d(in_channels=in_size, out_channels=out_size, kernel_size=4, stride=2, padding=1)
            self.conv_3 = nn.Conv1d(in_channels=in_size, out_channels=out_size, kernel_size=8, stride=2, padding=3)
            self.non_linearity = nn.ReLU

        def forward(self, x):
            # Transpose to  N x C x L
            x = torch.transpose(x, 1, 2)
            x = torch.cat([self.conv_1(x), self.conv_2(x), self.conv_3(x)], dim=1)
            x = self.non_linearity()(x)
            # Transpose back to N x L x C
            x = torch.transpose(x, 1, 2)
            return x

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.inception_block_1 = ChronoNet.InceptionBlock(self.input_size, out_size=32)
        self.inception_block_2 = ChronoNet.InceptionBlock(32*3, out_size=32)
        self.inception_block_3 = ChronoNet.InceptionBlock(32*3, out_size=32)

        cell = RnnBase.cell_mapper[self.rnn_cell_type]
        self.rnn = RNN(cell=cell, in_size=32*3, hidden_size=self.rnn_hidden_size,
                       num_layers=self.rnn_num_layers, dropout=self.dropout, batch_norm=self.batch_norm,
                       skip_mode=self.skip_mode)

        self.fc = nn.Linear(in_features=self.rnn_hidden_size, out_features=self.output_size, bias=True)

    def forward(self, x, hidden, context):
        x = self.inception_block_1(x)
        x = self.inception_block_2(x)
        x = self.inception_block_3(x)

        x, hidden = self.rnn(x, hidden)
        x = self.fc(x)

        return x, hidden

    def offset_size(self, sequence_size):
        assert sequence_size % 8 == 0,  "For this model it is better if sequence size is divisible by 8"
        return sequence_size - sequence_size//8


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