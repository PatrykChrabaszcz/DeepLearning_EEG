from src.deep_learning.pytorch.models.utils import LockedDropout, RNN, RNNLayerNorm
from src.deep_learning.pytorch.models.model_base import RnnBase
import torch.nn.functional as f
import torch.nn as nn
import logging
import torch


logger = logging.getLogger(__name__)


class SimpleRNN(RnnBase):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_dropout_layer = LockedDropout(self.dropout_i, use_mc_dropout=self.use_mc_dropout)

        cell = RnnBase.cell_mapper[self.rnn_cell_type]

        # Main RNN module
        self.rnn = RNN(cell=cell, in_size=self.input_size, hidden_size=self.rnn_hidden_size,
                       num_layers=self.rnn_num_layers, dropout_f=self.dropout_f, dropout_h=self.dropout_h,
                       rnn_normalization=self.rnn_normalization, dilation=self.rnn_dilation, skip_mode=self.skip_mode,
                       skip_first=self.skip_first, skip_last=self.skip_last, use_mc_dropout=self.use_mc_dropout)

        out_hidden_size = self.rnn_hidden_size if not self.use_context else self.rnn_hidden_size + self.context_size
        self.fc = nn.Linear(in_features=out_hidden_size, out_features=self.rnn_hidden_size, bias=True)
        self.fc2 = nn.Linear(in_features=self.rnn_hidden_size, out_features=self.output_size, bias=True)

    def forward(self, x, hidden, context):
        batch_size = x.size(0)
        time_size = x.size(1)

        # Dropout on the input features
        x = self.input_dropout_layer(x)

        # Rnn with all the features
        lstm_out, hidden = self.rnn(x, hidden)
        lstm_out = lstm_out.contiguous()

        if self.use_context:
            context = torch.cat([context] * lstm_out.size()[1], dim=1)
            lstm_out = torch.cat([lstm_out, context], dim=2)

        fc_out = self.fc(lstm_out.view(batch_size*time_size, lstm_out.size(2)))
        fc_out = f.relu(fc_out)
        fc_out = self.fc2(fc_out)

        fc_out = fc_out.view(batch_size, time_size, fc_out.size(1))

        return fc_out, hidden

    def offset_size(self, sequence_size):
        return 0
