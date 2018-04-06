from src.deep_learning.pytorch.models.rnn import RNNExtendedCell
from torch import cat, transpose
from torch.nn import Module
import logging

logger = logging.getLogger(__name__)


class RNN(Module):
    """
    Simply stacks RNNBlocks to form multilayer RNN according to specified settings. Implements dilation between
    the layers as described in the paper: "Dilated Recurrent Neural Networks" Chang et al..
    """
    def __init__(self, cell, in_size, hidden_size, num_layers, dropout_f=0.0, dropout_h=0.0,
                 rnn_normalization='none', dilation=1, skip_mode='none', skip_first=False, skip_last=False,
                 use_mc_dropout=False):
        super().__init__()
        assert dilation >= 1, 'Dilation (%s) has to be a positive integer' % dilation

        if skip_mode == 'none' and (skip_first or skip_last):
            logger.warning('Using skip_first (%s) or skip_last (%s) while skip_mode is none.'
                           'Defaults to not using skip_connections' % (skip_first, skip_last))
            skip_first = 0
            skip_last = 0
        if num_layers == 1 and skip_mode != 'none' and (skip_first != skip_last):
            logger.warning('For 1 layer skip_first (%s) and skip_last (%s) should be the same. '
                           'Defaults to using skip connection' % (skip_first, skip_last))
            skip_first = 1
            skip_last = 1

        self.dilation = dilation
        self.rnns = []
        for i in range(num_layers):
            # Do not apply skip connection in the first and in the last layer of the RNN if specified
            updated_skip_mode = skip_mode
            if skip_mode != 'none':
                if i == 0 and skip_first == 0:
                    updated_skip_mode = 'none'
                elif i == (num_layers-1) and skip_last == 0:
                    updated_skip_mode = 'none'
                else:
                    updated_skip_mode = skip_mode

            rnn = RNNExtendedCell(cell=cell, in_size=in_size, hidden_size=hidden_size, dropout_f=dropout_f,
                                  dropout_h=dropout_h, rnn_normalization=rnn_normalization,
                                  skip_mode=updated_skip_mode, use_mc_dropout=use_mc_dropout)
            self.add_module('rnn_block_%d' % i, rnn)
            self.rnns.append(rnn)

            if i == 0 and updated_skip_mode != 'concat':
                in_size = hidden_size
            elif updated_skip_mode == 'concat':
                in_size += hidden_size

    def forward(self, x, hidden):
        batch_size = x.size(0)
        time_size = x.size(1)

        out_hidden = []
        for layer_i, cell in enumerate(self.rnns):

            if layer_i != 0:
                x = [x[:, i::self.dilation, :] for i in range(self.dilation)]
                x = cat(x, dim=1)
                x = x.view(x.size(0)*self.dilation, x.size(1)//self.dilation, x.size(2))

            x, h = cell(x, hidden[layer_i])
            out_hidden.append(h)

        dilation = self.dilation ** (len(self.rnns) - 1)
        blocks = [x[i*dilation:(i+1)*dilation, :, :] for i in range(batch_size)]
        blocks = [transpose(b, 0, 1).contiguous().view(1, time_size, b.size(2)) for b in blocks]

        output = cat(blocks)

        return output, out_hidden
