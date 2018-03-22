from torch import nn, transpose, cat
import logging
import torch.nn.functional as f
from torch.nn import Parameter
import torch
from torch.autograd import Variable

# This module will look at signal of the format NumBatches x SequenceLength x Channels
# Transform it into NumBatches x Channels x SequenceLength
# Apply 1D convolution separately for each Channel
# Transform output signal back to NumBatches x SequenceLength x Channels


logger = logging.getLogger(__name__)


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


class RNNBatchNorm(nn.Module):
    def __init__(self, num_features):
        super().__init__()

        self.bn = nn.BatchNorm1d(num_features)
        self.bn.weight.data.fill_(1)

    def forward(self, x):
        x.contiguous()
        n, t = x.size(0), x.size(1)
        x = x.view(n * t, -1)
        x = self.bn(x)
        x = x.view(n, t, -1)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.bn.__repr__()
        tmpstr += ')'
        return tmpstr


class RNNLayerNorm(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.eps = 1e-5

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class RNNBlock(nn.Module):
    """
    Basic building block of RNN network.
    Implements:
    - forward dropout,
    - hidden to hidden weight matrix dropout,
    - batch and layer normalization
    - add and concat skip connections
    """
    def __init__(self, cell, in_size, hidden_size, dropout_f, dropout_h, rnn_normalization, skip_mode, use_mc_dropout):
        super().__init__()
        self.skip_mode = skip_mode

        # If dimensions not match we need to pass it through a cnn layer
        if skip_mode == 'add' and in_size != hidden_size:
            self.conv = nn.Conv1d(in_channels=in_size, out_channels=hidden_size, kernel_size=1, stride=1)
        else:
            self.conv = None

        # Pass through the RNN cell
        rnn = cell(input_size=in_size, hidden_size=hidden_size, num_layers=1, dropout=0, batch_first=True,
                   bias=True)

        # Apply dropout on the hidden/hidden weight matrix
        self.rnn = WeightDrop(module=rnn, weights=['weight_hh_l0'], dropout_h=dropout_h, use_mc_dropout=use_mc_dropout)

        # Apply batch normalization
        if rnn_normalization == 'batch_norm':
            self.normalization_layer = RNNBatchNorm(num_features=hidden_size)
        elif rnn_normalization == 'layer_norm':
            self.normalization_layer = RNNLayerNorm(num_features=hidden_size)
        elif rnn_normalization == 'none':
            self.normalization_layer = None
        else:
            raise RuntimeError('Unexpected rnn normalization %s' % rnn_normalization)

        # Apply the same dropout mask in each timestamp for the output
        self.dropout_layer = LockedDropout(dropout=dropout_f, use_mc_dropout=use_mc_dropout)

    def forward(self, x, hidden):
        x_skip = x
        x, hidden = self.rnn(x, hidden)

        if self.normalization_layer is not None:
            x = self.normalization_layer(x)

        x = self.dropout_layer(x)

        if self.skip_mode == 'concat':
            x = cat([x, x_skip], dim=2)
        elif self.skip_mode == 'add':
            if self.conv is not None:
                # Assume input comes as N x L x C
                # Transpose to  N x C x L
                x_skip = transpose(x_skip, 1, 2)
                x_skip = self.conv(x_skip)
                # Transpose back to N x L x C
                x_skip = transpose(x_skip, 1, 2)
            x = x + x_skip

        return x, hidden


class RNN(nn.Module):
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

            rnn = RNNBlock(cell=cell, in_size=in_size, hidden_size=hidden_size, dropout_f=dropout_f,
                           dropout_h=dropout_h, rnn_normalization=rnn_normalization, skip_mode=updated_skip_mode,
                           use_mc_dropout=use_mc_dropout)
            self.add_module('rnn_block_%d' % i, rnn)
            self.rnns.append(rnn)

            if i == 0:
                in_size = hidden_size
            elif skip_mode == 'concat':
                in_size += hidden_size

    def forward(self, input, hidden):
        batch_size = input.size(0)
        time_size = input.size(1)

        out_hidden = []
        for layer_i, cell in enumerate(self.rnns):

            if layer_i != 0:
                input = [input[:, i::self.dilation, :] for i in range(self.dilation)]
                input = torch.cat(input, dim=1)
                input = input.view(input.size(0)*self.dilation, input.size(1)//self.dilation, input.size(2))

            input, h = cell(input, hidden[layer_i])
            out_hidden.append(h)

        dilation = self.dilation ** (len(self.rnns) - 1)
        blocks = [input[i*dilation:(i+1)*dilation, :, :] for i in range(batch_size)]
        blocks = [torch.transpose(b, 0, 1).contiguous().view(1, time_size, b.size(2)) for b in blocks]

        output = torch.cat(blocks)

        return output, out_hidden


class IndGRU(nn.Module):
    def __init__(self, hidden_size, *args, **kwargs):
        super().__init__()
        self.module = nn.GRU(hidden_size=hidden_size, *args, **kwargs)

        # Terrible temporary solution to an issue regarding compacting weights re: CUDNN RNN
        # I'm not sure what is going on here, this is what weight_drop does so I stick to it
        self.module.flatten_parameters = self.widget_demagnetizer_y2k_edition

        # We need to register it in this module to make it work with weight dropout
        w_hh = torch.FloatTensor(3, hidden_size).type_as(getattr(self.module, 'weight_hh_l0').data)
        w_hh.uniform_(-1, 1)
        getattr(self.module, 'bias_ih_l0').data.fill_(0)
        getattr(self.module, 'bias_hh_l0').data.fill_(0)

        self.register_parameter(name='weight_hh_l0', param=Parameter(w_hh))
        del self.module._parameters['weight_hh_l0']

    def widget_demagnetizer_y2k_edition(*args, **kwargs):
        # We need to replace flatten_parameters with a nothing function
        # It must be a function rather than a lambda as otherwise pickling explodes
        # We can't write boring code though, so ... WIDGET DEMAGNETIZER Y2K EDITION!
        # (╯°□°）╯︵ ┻━┻
        return

    def _setweights(self):
        w_hh = getattr(self, 'weight_hh_l0')
        w_hr = torch.diag(w_hh[0, :])
        w_hz = torch.diag(w_hh[1, :])
        w_hn = torch.diag(w_hh[2, :])
        setattr(self.module, 'weight_hh_l0', torch.cat([w_hr, w_hz, w_hn], dim=1))

    def forward(self, *args):
        self._setweights()
        return self.module.forward(*args)


# Independent RNN as in the recent paper
class IndRNN(nn.Module):
    def __init__(self, hidden_size, *args, **kwargs):
        super().__init__()
        self.module = nn.RNN(hidden_size=hidden_size, *args, **kwargs, nonlinearity='relu')

        # Terrible temporary solution to an issue regarding compacting weights re: CUDNN RNN
        # I'm not sure what is going on here, this is what weight_drop does so I stick to it
        self.module.flatten_parameters = self.widget_demagnetizer_y2k_edition

        # We need to register it in this module to make it work with weight dropout
        w_hh = torch.FloatTensor(hidden_size).type_as(getattr(self.module, 'weight_hh_l0').data)
        w_hh.uniform_(-1, 1)

        getattr(self.module, 'bias_ih_l0').data.fill_(0)
        getattr(self.module, 'bias_hh_l0').data.fill_(0)

        self.register_parameter(name='weight_hh_l0', param=Parameter(w_hh))
        del self.module._parameters['weight_hh_l0']

    def widget_demagnetizer_y2k_edition(*args, **kwargs):
        # We need to replace flatten_parameters with a nothing function
        # It must be a function rather than a lambda as otherwise pickling explodes
        # We can't write boring code though, so ... WIDGET DEMAGNETIZER Y2K EDITION!
        # (╯°□°）╯︵ ┻━┻
        return

    def _setweights(self):
        w_hh = getattr(self, 'weight_hh_l0')
        w_hh = torch.diag(w_hh)
        setattr(self.module, 'weight_hh_l0', w_hh)

    def forward(self, *args):
        self._setweights()
        return self.module.forward(*args)

#
# class IndRNNCell(nn.Module):
#     """
#     IndRNN Cell
#     Performs a single time step operation
#     """
#     def __init__(self, inpdim, recdim):
#         super().__init__()
#         import math
#         self.inpdim = inpdim
#         self.recdim = recdim
#         self.act = f.relu
#         self.w = nn.Parameter(torch.zeros(inpdim, recdim))
#         self.w.data.normal_(0, math.sqrt(2. / (recdim + inpdim)))
#         self.u = nn.Parameter(torch.ones(recdim))
#         self.b = nn.Parameter(torch.zeros(recdim))
#         self.F = nn.Linear(recdim, 1)
#
#     def forward(self, x_t, h_tm1):
#         self.u.data.clamp(-1, 1)
#         return self.act(h_tm1 * self.u + x_t @ self.w + self.b)
#
#
# class IndRNN(nn.Module):
#     """
#     IndRNN
#     Given an input sequence, converts it to an output sequence.
#     """
#     def __init__(self, input_size, hidden_size, num_layers=1, dropout=0, batch_first=True,
#                  bias=True):
#         """
#         inpdim      : dimension D in (Batch, Time, D)
#         recdim      : recurrent dimension/ Units/
#         depth       : stack depth
#         """
#         super().__init__()
#         self.inpdim = input_size
#         self.recdim = hidden_size
#         self.cell = IndRNNCell(input_size, hidden_size)
#
#     def forward(self, x, hidden):
#         h_tm1 = hidden
#         seq = []
#         for i in range(x.size()[1]):
#             x_t = x[:, i, :]
#             h_tm1 = self.cell.forward(x_t, h_tm1)
#             seq.append(h_tm1)
#         seq = torch.squeeze(torch.stack(seq, dim=2))
#
#         return seq, h_tm1


# https://github.com/salesforce/awd-lstm-lm/blob/master/weight_drop.py
class WeightDrop(nn.Module):
    """
    CuDNN implementation of RNN networks is much faster but also limited. We are not able to specify the dropout
    on hidden to hidden connections. If we use an implementation that allow to do that we will lose a lot on speed.
    As a solution we use DropConnect on Hidden to Hidden matrices. This will apply the same dropout mask for
    every timepoint and every example within the minibatch.
    """
    def __init__(self, module, weights, dropout_h=0, use_mc_dropout=False):
        super().__init__()
        self.module = module
        self.weights = weights
        self.dropout_h = dropout_h
        self.use_mc_dropout = use_mc_dropout

        # If dropout_h is set to 0 then only call module in forward
        if self.dropout_h != 0:
            self._setup()

    def widget_demagnetizer_y2k_edition(*args, **kwargs):
        # We need to replace flatten_parameters with a nothing function
        # It must be a function rather than a lambda as otherwise pickling explodes
        # We can't write boring code though, so ... WIDGET DEMAGNETIZER Y2K EDITION!
        # (╯°□°）╯︵ ┻━┻
        return

    def _setup(self):
        # Terrible temporary solution to an issue regarding compacting weights re: CUDNN RNN
        if issubclass(type(self.module), nn.RNNBase):
            self.module.flatten_parameters = self.widget_demagnetizer_y2k_edition

        for name_w in self.weights:
            w = getattr(self.module, name_w)
            del self.module._parameters[name_w]
            self.module.register_parameter(name_w + '_raw', Parameter(w.data))

    def _setweights(self):
        for name_w in self.weights:
            raw_w = getattr(self.module, name_w + '_raw')
            if self.use_mc_dropout:
                w = f.dropout(raw_w, p=self.dropout_h, training=True)
            else:
                w = f.dropout(raw_w, p=self.dropout_h, training=self.training)
            setattr(self.module, name_w, w)

    def forward(self, *args):
        if self.dropout_h != 0:
            self._setweights()

        return self.module.forward(*args)


# https://github.com/salesforce/awd-lstm-lm/blob/master/locked_dropout.py
class LockedDropout(nn.Module):
    """
    We want to use the same dropout mask for all timepoints. Using this layer we will be able to do so. Dropout masks
    will be different for different examples within the minibatch but will not change in timesteps.
    """
    def __init__(self, dropout, use_mc_dropout):
        super().__init__()
        assert 0.0 <= dropout <= 1.0, 'Dropout has to be in range <0.0, 1.0>'
        self.use_mc_dropout = use_mc_dropout
        self.dropout = dropout

    def forward(self, x):
        if (self.training or self.use_mc_dropout) and self.dropout != 0:
            # Same dropout for all timesteps
            m = x.data.new(x.size(0), 1, x.size(2)).bernoulli_(1.0 - self.dropout)

            # Handle the case where dropout is 1.0 (Should act as blocking information flow)
            mask = Variable(m, requires_grad=False)
            if self.dropout != 1.0:
                mask /= (1.0 - self.dropout)
            return mask * x
        else:
            return x


if __name__ == '__main__':
    import numpy as np
    batch = np.array([
        [[1, 1, 1], [2, 2, 2], [3, 3, 4]],
        [[1, 1, 1], [2, 2, 2], [3, 3, 3]]]).astype(np.float32)

    batch = Variable(torch.from_numpy(batch))

    d = RNNBatchNorm(3)

    l = RNNLayerNorm(3)
    print([p for p in d.parameters()])
    print('Input batch')
    print(batch)

    print('Batch Norm output')
    print(d(batch))

    print('Layer Norm output')
    print(l(batch))





