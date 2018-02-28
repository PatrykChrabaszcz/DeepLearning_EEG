from torch import nn, cat, unsqueeze, transpose, cat

from torch.nn import Parameter
from functools import wraps
import torch
from torch.autograd import Variable


# This module will look at signal of the format NumBatches x SequenceLength x Channels
# Transform it into NumBatches x Channels x SequenceLength
# Apply 1D convolution separately for each Channel
# Transform output signal back to NumBatches x SequenceLength x Channels

class SeparateChannelCNN(nn.Module):
    def __init__(self, in_size, out_size, num_layers, kernel_size, stride, batch_norm=False):
        assert out_size % in_size == 0

        super().__init__()
        groups = in_size

        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.stride = stride

        self.cnns = []
        self.batch_norm = batch_norm
        self.bnns = []
        for i in range(num_layers):
            # Convolution layers
            cnn = nn.Conv1d(in_channels=in_size, out_channels=out_size,
                            kernel_size=kernel_size, stride=stride, groups=groups, bias=not batch_norm)
            in_size = out_size
            self.add_module('cnn_c_%s' % i, cnn)
            self.cnns.append(cnn)
            # Batch norm layers
            if batch_norm:
                bnn = nn.BatchNorm1d(out_size)
                self.add_module('bnn_c_%s' % i, bnn)
                self.bnns.append(bnn)

        # Non-linearity layer
        self.non_linearity = nn.ReLU

    def forward(self, x):
        # Assume input comes as N x L x C
        # Transpose to  N x C x L
        x = transpose(x, 1, 2)
        for i, conv in enumerate(self.cnns):
            x = conv(x)
            x = self.non_linearity()(x)
            if self.batch_norm:
                x = self.bnns[i](x)

        # Transpose back to N x L x C
        x = transpose(x, 1, 2)
        return x

    def out_seq_size(self, seq_size):
        for i in range(self.num_layers):
            seq_size = (seq_size - self.kernel_size) // self.stride + 1
        return seq_size


class AllChannelsCNN(nn.Module):
    def __init__(self, in_size, out_size, num_layers, kernel_size, stride, batch_norm=False):
        super().__init__()

        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.stride = stride

        self.cnns = []
        self.batch_norm = batch_norm
        self.bnns = []
        for i in range(num_layers):
            # Convolution layers
            cnn = nn.Conv1d(in_channels=in_size, out_channels=out_size,
                            kernel_size=kernel_size, stride=stride, bias=not batch_norm)
            in_size = out_size
            self.add_module('cnn_c_%s' % i, cnn)
            self.cnns.append(cnn)
            # Batch norm layers
            if batch_norm:
                bnn = nn.BatchNorm1d(out_size)
                self.add_module('bnn_c_%s' % i, bnn)
                self.bnns.append(bnn)

        # Non-linearity layer
        self.non_linearity = nn.ReLU

    def forward(self, x):
        # Assume input comes as N x L x C
        # Transpose to  N x C x L
        x = transpose(x, 1, 2)
        for i, conv in enumerate(self.cnns):
            x = conv(x)
            x = self.non_linearity()(x)
            if self.batch_norm:
                x = self.bnns[i](x)

        # Transpose back to N x L x C
        x = transpose(x, 1, 2)
        return x

    def out_seq_size(self, seq_size):
        for i in range(self.num_layers):
            seq_size = (seq_size - self.kernel_size) // self.stride + 1
        return seq_size


class QRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_first):
        super().__init__()
        from torchqrnn import QRNN as qrnn
        self.cell = qrnn(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=False)

    def forward(self, x, h):
        # Transpose to  L x N x C
        x = transpose(x, 0, 1).contiguous()
        x, h = self.cell(x, h)
        # Transpose back to L x N x C
        x = transpose(x, 0, 1)

        return x, h


class RNNBlock(nn.Module):
    def __init__(self, cell, in_size, hidden_size, dropout_f, dropout_h, batch_norm, skip_mode):
        super().__init__()
        out_size = hidden_size if skip_mode != 'concat' else hidden_size + in_size
        self.skip_mode = skip_mode

        # Pass through the RNN cell
        rnn = cell(input_size=in_size, hidden_size=hidden_size, num_layers=1, dropout=0, batch_first=True)

        # Apply dropout on the hidden/hidden weight matrix
        if dropout_h != 0.0:
            self.rnn = WeightDrop(module=rnn, weights=['weight_hh_l0'], dropout=dropout_h)
        else:
            self.rnn = rnn

        # Apply batch normalization
        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(num_features=out_size)
        else:
            self.batch_norm = None

        # Apply the same dropout mask in each timestamp for the output
        self.dropout_f = dropout_f
        if self.dropout_f != 0.0:
            self.dropout = LockedDropout()
        else:
            self.dropout = None

        # Add the end apply skip connection if required

    def forward(self, x, hidden):
        x_skip = x
        x, hidden = self.rnn(x, hidden)

        if self.batch_norm is not None:
            # Transpose to  N x C x L
            x = transpose(x, 1, 2).contiguous()
            x = self.batch_norm(x)
            # Transpose back to N x L x C
            x = transpose(x, 1, 2)

        if self.dropout_f != 0.0:
            x = self.dropout(x, self.dropout_f)

        if self.skip_mode == 'concat':
            x = cat([x, x_skip], dim=2)
        elif self.skip_mode == 'add':
            x = x + x_skip

        return x, hidden


class RNN(nn.Module):
    def __init__(self, cell, in_size, hidden_size, num_layers, dropout_f=0.0, dropout_h=0.0,
                 batch_norm=True, skip_mode='none'):
        super().__init__()

        self.rnns = []
        for i in range(num_layers):
            # Do not apply skip connection in the first and in the last layer of the RNN
            if i == 0 or i == num_layers - 1:
                skip = 'none'
            else:
                skip = skip_mode

            rnn = RNNBlock(cell, in_size, hidden_size, dropout_f, dropout_h, batch_norm, skip_mode=skip)
            self.add_module('rnn_block_%d' % i, rnn)

            self.rnns.append(rnn)

            if i == 0:
                in_size = hidden_size
            elif skip_mode == 'concat':
                in_size += hidden_size

    def forward(self, x, h):
        h_last = []

        for i, rnn in enumerate(self.rnns):
            if isinstance(h, tuple):
                h_curr = (h_[i].unsqueeze(0) for h_ in h)
            elif h is not None:
                h_curr = h[i].unsqueeze(0)
            else:
                h_curr = h
            x, h_layer = rnn(x, h_curr)

            h_last.append(h_layer)

        return x, cat(h_last)


# # In a full setting (when everything is used)
# # A = Input
# # B = RNN Layer(A)
# # C = Batch Norm(B)
# # D = Dropout(C)
# # F = RNN Layer (E)
# # G = Residual (F, D)
# # H = Batch Norm(F)
# # I = Dropout(H)
# class RNN(nn.Module):
#     def __init__(self, cell, in_size, hidden_size, num_layers, dropout=0.0, batch_norm=True, skip_mode='none'):
#         super().__init__()
#
#         self.skip_mode = skip_mode
#
#         self.rnns = []
#         self.batch_norm = batch_norm
#         self.bnns = []
#
#         # Not sure if this first skip mode should be there or not? Maybe a parameter ??
#         # If skip mode is 'add' and input size is different than hidden size then we will need additional conv layer to
#         # match dimensions
#         # self.skip_layer = nn.Conv1d(in_channels=in_size, out_channels=hidden_size, kernel_size=1, stride=1) \
#         #     if in_size != hidden_size and skip_mode == 'add' else None
#         self.skip_layer = None
#
#         # Note that this is not optimal for standard LSTM,GRU and num_layers > 1
#         # But we do not want to mess up with 1000 different implementations so we provide one that can handle
#         # all the stuff at the cost of handling some stuff slower
#         for i in range(num_layers):
#             rnn = cell(input_size=in_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
#             self.add_module('rnn_%s' % i, rnn)
#             self.rnns.append(rnn)
#             in_size = hidden_size if skip_mode != 'concat' else hidden_size + in_size
#             if batch_norm:
#                 bnn = nn.BatchNorm1d(num_features=hidden_size)
#                 self.add_module('bnn_c_%s' % i, bnn)
#                 self.bnns.append(bnn)
#
#         self.dropout_layer = nn.Dropout(p=dropout) if dropout > 0.0 else None
#
#     def forward(self, x, h):
#         h_last = []
#
#         if self.skip_layer:
#             # Transpose to  N x C x L
#             x_skip = transpose(x, 1, 2).contiguous()
#             x_skip = self.skip_layer(x_skip)
#             # Transpose back to N x L x C
#             x_skip = transpose(x_skip, 1, 2)
#         else:
#             x_skip = x
#
#         for i, rnn in enumerate(self.rnns):
#             if isinstance(h, tuple):
#                 h_curr = (h_[i].unsqueeze(0) for h_ in h)
#             elif h is not None:
#                 h_curr = h[i].unsqueeze(0)
#             else:
#                 h_curr = h
#             x, h_layer = rnn(x, h_curr)
#
#             if self.batch_norm:
#                 # Transpose to  N x C x L
#                 x = transpose(x, 1, 2).contiguous()
#                 x = self.bnns[i](x)
#                 # Transpose back to N x L x C
#                 x = transpose(x, 1, 2)
#
#             h_last.append(h_layer)
#
#             if self.skip_mode == 'add' and i != len(self.rnns) - 1:
#                 x += x_skip
#                 x_skip = x
#
#             elif self.skip_mode == 'concat' and i != len(self.rnns) - 1:
#                 x = cat([x, x_skip], dim=2)
#                 x_skip = x
#
#             if self.dropout_layer:
#                 x = self.dropout_layer(x)
#
#         return x, cat(h_last)

#
# class BNRNN(nn.Module):
#     def __init__(self, cell_type, input_size, hidden_size, num_layers=5, batch_first=True, dropout=0.0):
#         super(BNRNN, self).__init__()
#
#         self.num_layers = num_layers
#         self.dropout = dropout
#
#         self.rnns = []
#
#         rnn = BatchRNNCell(cell_type=cell_type, input_size=input_size, hidden_size=hidden_size, batch_norm=False)
#         rnn.flatten_parameters()
#
#         self.add_module('rnn_0', rnn)
#         self.rnns.append(rnn)
#         self.dropouts = []
#
#         for l in range(1, num_layers):
#             dropout_layer = nn.Dropout(p=dropout)
#             self.add_module('dropout_%d' % l, dropout_layer)
#             self.dropouts.append(dropout_layer)
#
#             rnn = BatchRNNCell(cell_type=cell_type, input_size=hidden_size, hidden_size=hidden_size)
#             rnn.flatten_parameters()
#             self.add_module('rnn_%d' % l, rnn)
#             self.rnns.append(rnn)
#
#     def forward(self, x, h):
#         h_last = []
#         for i, rnn in enumerate(self.rnns):
#
#             if isinstance(h, tuple):
#                 h_curr = (h_[i].unsqueeze(0) for h_ in h)
#             else:
#                 h_curr = h[i].unsqueeze(0)
#             x, h_layer = rnn(x, h_curr)
#
#             if i < len(self.dropouts):
#                 x = self.dropouts[i](x)
#
#             h_last.append(h_layer)
#
#         return x, cat(h_last)
#
#
# class BNLSTM(BNRNN):
#     def __init__(self, input_size, hidden_size, num_layers, batch_first=True, dropout=0.0):
#         super(BNLSTM, self).__init__(nn.LSTM, input_size, hidden_size, num_layers, dropout=dropout)
#
#
# class BNGRU(BNRNN):
#     def __init__(self, input_size, hidden_size, num_layers, batch_first=True, dropout=0.0):
#         super(BNGRU, self).__init__(nn.GRU, input_size, hidden_size, num_layers, dropout=dropout)





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


# https://github.com/salesforce/awd-lstm-lm/blob/master/weight_drop.py
class WeightDrop(torch.nn.Module):
    def __init__(self, module, weights, dropout=0, variational=False):
        super(WeightDrop, self).__init__()
        self.module = module
        self.weights = weights
        self.dropout = dropout
        self.variational = variational
        self._setup()

    def widget_demagnetizer_y2k_edition(*args, **kwargs):
        # We need to replace flatten_parameters with a nothing function
        # It must be a function rather than a lambda as otherwise pickling explodes
        # We can't write boring code though, so ... WIDGET DEMAGNETIZER Y2K EDITION!
        # (╯°□°）╯︵ ┻━┻
        return

    def _setup(self):
        # Terrible temporary solution to an issue regarding compacting weights re: CUDNN RNN
        if issubclass(type(self.module), torch.nn.RNNBase):
            self.module.flatten_parameters = self.widget_demagnetizer_y2k_edition

        for name_w in self.weights:
            print('Applying weight drop of {} to {}'.format(self.dropout, name_w))
            w = getattr(self.module, name_w)
            del self.module._parameters[name_w]
            self.module.register_parameter(name_w + '_raw', Parameter(w.data))

    def _setweights(self):
        for name_w in self.weights:
            raw_w = getattr(self.module, name_w + '_raw')
            w = None
            if self.variational:
                mask = torch.autograd.Variable(torch.ones(raw_w.size(0), 1))
                if raw_w.is_cuda: mask = mask.cuda()
                mask = torch.nn.functional.dropout(mask, p=self.dropout, training=True)
                w = mask.expand_as(raw_w) * raw_w
            else:
                w = torch.nn.functional.dropout(raw_w, p=self.dropout, training=self.training)
            setattr(self.module, name_w, w)

    def forward(self, *args):
        self._setweights()
        return self.module.forward(*args)

# https://github.com/salesforce/awd-lstm-lm/blob/master/locked_dropout.py
# But we use batch_first=True so we need to change it a little bit
class LockedDropout(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, dropout=0.5):
        if not self.training or not dropout:
            return x
        #m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
        m = x.data.new(x.size(0), 1, x.size(2)).bernoulli_(1 - dropout)
        mask = Variable(m, requires_grad=False) / (1 - dropout)
        mask = mask.expand_as(x)
        return mask * x


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