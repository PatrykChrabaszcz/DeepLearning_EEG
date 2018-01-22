from torch import nn, cat, unsqueeze
from collections import OrderedDict


#https://github.com/SeanNaren/deepspeech.pytorch/blob/master/model.py
class SequenceWise(nn.Module):
    def __init__(self, module):
        """
        Collapses input of dim T*N*H to (T*N)*H, and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.
        :param module: Module to apply input to.
        """
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x):
        t, n = x.size(0), x.size(1)
        x = x.view(t * n, -1)
        x = self.module(x)
        x = x.view(t, n, -1)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr


class BatchRNNCell(nn.Module):
    def __init__(self, cell_type, input_size, hidden_size, batch_norm=True):
        super(BatchRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_norm = SequenceWise(nn.BatchNorm1d(input_size)) if batch_norm else None

        self.rnn = cell_type(input_size=input_size, hidden_size=hidden_size, num_layers=1,
                             batch_first=True, bias=False)

    def flatten_parameters(self):
        self.rnn.flatten_parameters()

    def forward(self, x, h):
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        x, h = self.rnn(x, h)
        return x, h


class BNRNN(nn.Module):
    def __init__(self, cell_type, input_size, hidden_size, num_layers=5, batch_first=True, dropout=0.0):
        super(BNRNN, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        self.rnns = []

        rnn = BatchRNNCell(cell_type=cell_type, input_size=input_size, hidden_size=hidden_size, batch_norm=False)
        rnn.flatten_parameters()

        self.add_module('rnn_0', rnn)
        self.rnns.append(rnn)
        self.dropouts = []

        for l in range(1, num_layers):
            dropout_layer = nn.Dropout(p=dropout)
            self.add_module('dropout_%d' % l, dropout_layer)
            self.dropouts.append(dropout_layer)

            rnn = BatchRNNCell(cell_type=cell_type, input_size=hidden_size, hidden_size=hidden_size)
            rnn.flatten_parameters()
            self.add_module('rnn_%d' % l, rnn)
            self.rnns.append(rnn)

    def forward(self, x, h):
        h_last = []
        for i, rnn in enumerate(self.rnns):

            if isinstance(h, tuple):
                h_curr = (h_[i].unsqueeze(0) for h_ in h)
            else:
                h_curr = h[i].unsqueeze(0)
            x, h_layer = rnn(x, h_curr)

            if i < len(self.dropouts):
                x = self.dropouts[i](x)

            h_last.append(h_layer)

        return x, cat(h_last)


class BNLSTM(BNRNN):
    def __init__(self, input_size, hidden_size, num_layers, batch_first=True, dropout=0.0):
        super(BNLSTM, self).__init__(nn.LSTM, input_size, hidden_size, num_layers, dropout=dropout)


class BNGRU(BNRNN):
    def __init__(self, input_size, hidden_size, num_layers, batch_first=True, dropout=0.0):
        super(BNGRU, self).__init__(nn.GRU, input_size, hidden_size, num_layers, dropout=dropout)

#https://github.com/pytorch/pytorch/issues/1959

# class LayerNormGRUCell(nn.GRUCell):
#     def __init__(self, input_size, hidden_size, bias=True):
#         super(LayerNormGRUCell, self).__init__(input_size, hidden_size, bias)
#
#         self.gamma_ih = nn.Parameter(torch.ones(3 * self.hidden_size))
#         self.gamma_hh = nn.Parameter(torch.ones(3 * self.hidden_size))
#         self.eps = 0
#
#     def _layer_norm_x(self, x, g, b):
#         mean = x.mean(1).expand_as(x)
#         std = x.std(1).expand_as(x)
#         return g.expand_as(x) * ((x - mean) / (std + self.eps)) + b.expand_as(x)
#
#     def _layer_norm_h(self, x, g, b):
#         mean = x.mean(1).expand_as(x)
#         return g.expand_as(x) * (x - mean) + b.expand_as(x)
#
#     def forward(self, x, h):
#
#         ih_rz = self._layer_norm_x(
#             torch.mm(x, self.weight_ih.narrow(0, 0, 2 * self.hidden_size).transpose(0, 1)),
#             self.gamma_ih.narrow(0, 0, 2 * self.hidden_size),
#             self.bias_ih.narrow(0, 0, 2 * self.hidden_size))
#
#         hh_rz = self._layer_norm_h(
#             torch.mm(h, self.weight_hh.narrow(0, 0, 2 * self.hidden_size).transpose(0, 1)),
#             self.gamma_hh.narrow(0, 0, 2 * self.hidden_size),
#             self.bias_hh.narrow(0, 0, 2 * self.hidden_size))
#
#         rz = torch.sigmoid(ih_rz + hh_rz)
#         r = rz.narrow(1, 0, self.hidden_size)
#         z = rz.narrow(1, self.hidden_size, self.hidden_size)
#
#         ih_n = self._layer_norm_x(
#             torch.mm(x, self.weight_ih.narrow(0, 2 * self.hidden_size, self.hidden_size).transpose(0, 1)),
#             self.gamma_ih.narrow(0, 2 * self.hidden_size, self.hidden_size),
#             self.bias_ih.narrow(0, 2 * self.hidden_size, self.hidden_size))
#
#         hh_n = self._layer_norm_h(
#             torch.mm(h, self.weight_hh.narrow(0, 2 * self.hidden_size, self.hidden_size).transpose(0, 1)),
#             self.gamma_hh.narrow(0, 2 * self.hidden_size, self.hidden_size),
#             self.bias_hh.narrow(0, 2 * self.hidden_size, self.hidden_size))
#
#         n = torch.tanh(ih_n + r * hh_n)
#         h = (1 - z) * n + z * h
#         return h
#
# class LayerNormGRU(nn.Module):
#     def __init__(self, input_size, hidden_size, bias=True):
#         super(LayerNormGRU, self).__init__()
#         self.cell = LayerNormGRUCell(input_size, hidden_size, bias)
#         self.weight_ih_l0 = self.cell.weight_ih
#         self.weight_hh_l0 = self.cell.weight_hh
#         self.bias_ih_l0 = self.cell.bias_ih
#         self.bias_hh_l0 = self.cell.bias_hh
#
#     def forward(self, xs, h):
#         h = h.squeeze(0)
#         ys = []
#         for i in range(xs.size(0)):
#             x = xs.narrow(0, i, 1).squeeze(0)
#             h = self.cell(x, h)
#             ys.append(h.unsqueeze(0))
#         y = torch.cat(ys, 0)
#         h = h.unsqueeze(0)
#         return y, h