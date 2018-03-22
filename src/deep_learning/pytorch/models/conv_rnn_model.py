from src.deep_learning.pytorch.models.utils import SeparateChannelCNN, AllChannelsCNN, RNN
from src.deep_learning.pytorch.models.model_base import RnnBase
import torch.nn as nn
import logging


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
        parser.add_argument("cnn_c_stride", type=int, default=2,
                            help="Stride for the first CNN block.")

        parser.add_argument("cnn_f_layers", type=int, default=3,
                            help="Number of layers in the second cnn block. "
                                 "All channels processed together")
        parser.add_argument("cnn_f_channels", type=int, default=10,
                            help="Number of filters in the second cnn block.")
        parser.add_argument("cnn_f_width", type=int, default=10,
                            help="Width in time dimension of the kernels in the second cnn block.")
        parser.add_argument("cnn_f_stride", type=int, default=2,
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
                       num_layers=self.rnn_num_layers, dropout_f=self.dropout, batch_norm=self.batch_norm)

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
