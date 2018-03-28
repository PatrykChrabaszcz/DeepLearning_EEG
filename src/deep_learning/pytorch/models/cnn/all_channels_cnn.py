from torch.nn import Module, Conv1d, BatchNorm1d, Sequential, ReLU
from torch import transpose


class AllChannelsCNN(Module):
    def __init__(self, in_size, out_size, num_layers, kernel_size, stride, input_in_rnn_format=False, batch_norm=False):
        super().__init__()

        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.stride = stride
        self.input_in_rnn_format = input_in_rnn_format
        self.sequential = Sequential()

        for i in range(num_layers):
            cnn = Conv1d(in_channels=in_size, out_channels=out_size,
                         kernel_size=kernel_size, stride=stride, bias=not batch_norm)
            self.sequential.add_module('cnn_c_%s' % i, cnn)
            self.sequential.add_module('non_lin_%s' % i, ReLU())
            if batch_norm:
                bnn = BatchNorm1d(out_size)
                self.sequential.add_module('bnn_c_%s' % i, bnn)
            in_size = out_size

    def forward(self, x):
        if self.input_in_rnn_format:
            # RNN format: N x L x C
            # Transpose to CNN format:  N x C x L
            x = transpose(x, 1, 2)

        x = self.sequential(x)

        if self.input_in_rnn_format:
            # RNN format: N x L x C
            # Transpose to CNN format:  N x C x L
            x = transpose(x, 1, 2)

        return x

    # Probably not necessary with a new trick (forwarding dummy tensor)
    def out_seq_size(self, seq_size):
        for i in range(self.num_layers):
            seq_size = (seq_size - self.kernel_size) // self.stride + 1
        return seq_size
