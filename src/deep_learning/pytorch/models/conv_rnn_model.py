from src.deep_learning.pytorch.models.cnn import SeparateChannelCNN, AllChannelsCNN
from src.deep_learning.pytorch.models.rnn import RNN, RNNLockedDropout
from src.deep_learning.pytorch.models.model_base import RnnBase
from torch.autograd import Variable
from torch import cat, from_numpy
import torch.nn as nn
import logging
import numpy as np


logger = logging.getLogger(__name__)


class ConvRNN(RnnBase):
    @staticmethod
    def add_arguments(parser):
        RnnBase.add_arguments(parser)
        parser.add_argument("cnn_c_layers", type=int, default=3,
                            help="Number of layers in the first cnn block. "
                                 "Each channel processed separately the same filters.")
        parser.add_argument("cnn_c_channels", type=int, default=10,
                            help="Number of filters in the first cnn block.")
        parser.add_argument("cnn_c_width", type=int, default=10,
                            help="Width in time dimension of the kernels in the first cnn block.")
        parser.add_argument("cnn_c_dilation", type=int, default=2,
                            help="Dilation for the first CNN block.")
        parser.add_argument("cnn_c_stride", type=int, default=2,
                            help="Stride for the first CNN block.")

        parser.add_argument("cnn_f_layers", type=int, default=3,
                            help="Number of layers in the second cnn block. "
                                 "All channels processed together")
        parser.add_argument("cnn_f_channels", type=int, default=10,
                            help="Number of filters in the second cnn block.")
        parser.add_argument("cnn_f_width", type=int, default=10,
                            help="Width in time dimension of the kernels in the second cnn block.")
        parser.add_argument("cnn_f_dilation", type=int, default=2,
                            help="Stride for the second CNN block.")
        parser.add_argument("cnn_f_stride", type=int, default=2,
                            help="Stride for the second CNN block.")

        parser.add_argument("cnn_batch_norm", type=int, default=0,
                            help="Stride for the second CNN block.")
        return parser

    def __init__(self, cnn_c_layers, cnn_c_channels, cnn_c_width, cnn_c_dilation, cnn_c_stride,
                 cnn_f_layers, cnn_f_channels, cnn_f_width, cnn_f_dilation, cnn_f_stride, cnn_batch_norm, **kwargs):

        super().__init__(**kwargs)
        self.input_dropout_layer = RNNLockedDropout(self.dropout_i, use_mc_dropout=self.use_mc_dropout)

        self.cnn_c = SeparateChannelCNN(in_size=self.input_size, out_size=cnn_c_channels, num_layers=cnn_c_layers,
                                        kernel_size=cnn_c_width, dilation=cnn_c_dilation, stride=cnn_c_stride,
                                        batch_norm=cnn_batch_norm, input_in_rnn_format=True)

        in_size = self.input_size * cnn_c_channels
        self.cnn_f = AllChannelsCNN(in_size=in_size, out_size=cnn_f_channels, num_layers=cnn_f_layers,
                                    kernel_size=cnn_f_width, dilation=cnn_f_dilation, stride=cnn_f_stride,
                                    batch_norm=cnn_batch_norm, input_in_rnn_format=True)

        cell = RnnBase.cell_mapper[self.rnn_cell_type]
        self.rnn = RNN(cell=cell, in_size=cnn_f_channels, hidden_size=self.rnn_hidden_size,
                       num_layers=self.rnn_num_layers, dropout_f=self.dropout_f, dropout_h=self.dropout_h,
                       rnn_normalization=self.rnn_normalization, dilation=self.rnn_dilation, skip_mode=self.skip_mode,
                       skip_first=self.skip_first, skip_last=self.skip_last, use_mc_dropout=self.use_mc_dropout)

        self.fc = nn.Linear(in_features=self.rnn_hidden_size, out_features=self.output_size, bias=True)

    def forward(self, x, hidden, context):
        batch_size = x.size(0)
        time_size = x.size(1)

        if self.use_context:
            # Repeat context for each time-step
            context = cat([context] * time_size, dim=1)
            x = cat([x, context], dim=2)
            assert list(x.size()) == [batch_size, time_size, self.input_size + self.context_size]

        # Dropout on the input features
        x = self.input_dropout_layer(x)

        # Conv layers
        x = self.cnn_c(x)
        x = self.cnn_f(x)

        # Rnn with extended cell
        lstm_out, hidden = self.rnn(x, hidden)
        lstm_out = lstm_out.contiguous()

        fc_out = self.fc(lstm_out.view(-1, lstm_out.size(2)))
        fc_out = fc_out.view(batch_size, -1, fc_out.size(1))

        return fc_out, hidden

    def offset_size(self, sequence_size):
        # Forward dummy vector and find out what is the output shape

        v = np.zeros((1, sequence_size, self.input_size), np.float32)
        v = Variable(from_numpy(v))

        c = np.zeros((1, self.context_size), np.float32)
        c = Variable(from_numpy(c))

        s = self.import_state([self.initial_state()])

        if next(self.parameters()).is_cuda:
            v = v.cuda()
            c = c.cuda()
            s = [_s.cuda() for _s in s]

        o, h = self.forward(v, s, c)
        o_size = o.size(1)

        return sequence_size - o_size
