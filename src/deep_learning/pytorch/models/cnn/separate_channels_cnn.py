from torch.nn import Module, BatchNorm1d, Conv1d, ReLU
from torch import transpose


class SeparateChannelCNN(Module):
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
            cnn = Conv1d(in_channels=in_size, out_channels=out_size,
                         kernel_size=kernel_size, stride=stride, groups=groups, bias=not batch_norm)
            in_size = out_size
            self.add_module('cnn_c_%s' % i, cnn)
            self.cnns.append(cnn)
            # Batch norm layers
            if batch_norm:
                bnn = BatchNorm1d(out_size)
                self.add_module('bnn_c_%s' % i, bnn)
                self.bnns.append(bnn)

        # Non-linearity layer
        self.non_linearity = ReLU

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

