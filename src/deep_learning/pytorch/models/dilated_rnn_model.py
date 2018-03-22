import torch
import torch.nn as nn
import torch.autograd as autograd
from src.deep_learning.pytorch.models.model_base import RnnBase
from torch.autograd import Variable
import numpy as np
use_cuda = torch.cuda.is_available()


class DilatedRnnModel(RnnBase):
    @staticmethod
    def add_arguments(parser):
        RnnBase.add_arguments(parser)
        return parser

    def __init__(self, **kwargs):
        super(DilatedRnnModel, self).__init__(**kwargs)

        self.dilations = [2 ** i for i in range(self.rnn_num_layers)]

        self.cells = torch.nn.ModuleList([])

        cell_type = self.cell_mapper[self.rnn_cell_type]
        for i in range(self.rnn_num_layers):
            if i == 0:
                c = cell_type(self.input_size, self.rnn_hidden_size, batch_first=True)
            else:
                c = cell_type(self.rnn_hidden_size, self.rnn_hidden_size, batch_first=True)
            self.cells.append(c)

        rnn_hidden_size = self.rnn_hidden_size if not self.use_context else self.rnn_hidden_size + self.context_size
        self.fc = nn.Linear(in_features=rnn_hidden_size, out_features=self.rnn_hidden_size, bias=True)
        self.fc2 = nn.Linear(in_features=self.rnn_hidden_size, out_features=self.output_size, bias=True)

    def forward(self, input, hidden, context):
        batch_size = input.size(0)
        time_size = input.size(1)
        out_hidden = []
        for i, cell in enumerate(self.cells):
            if i != 0:
                input = input[:, 0::2, :], input[:, 1::2, :]
                input = torch.cat(input, dim=1)
                input = input.view(input.size(0)*2, input.size(1)//2, input.size(2))

            input, h = cell(input, hidden[i])
            out_hidden.append(h)

        dilation = 2 ** (len(self.cells) - 1)
        blocks = [input[i*dilation:(i+1)*dilation, :, :] for i in range(batch_size)]
        blocks = [torch.transpose(b, 0, 1).contiguous().view(1, time_size, b.size(2)) for b in blocks]

        output = torch.cat(blocks)

        if self.use_context:
            context = torch.cat([context] * output.size()[1], dim=1)
            output = torch.cat([output, context], dim=2)

        fc_out = self.fc(output.view(output.size(0) * output.size(1), output.size(2)))
        fc_out = nn.ReLU()(fc_out)
        fc_out = self.fc2(fc_out)
        fc_out = fc_out.view(output.size(0), output.size(1), fc_out.size(1))

        return fc_out, out_hidden

    # Normally RNN would have the state as an array [layers, hidden_size] for each sample in the minibatch
    # For dilated RNN the state will be stored as [[1, hidden_size], [2, hidden_size], [4, hidden_size] ... ]
    def initial_state(self):
        if self.rnn_cell_type == "LSTM":
            c = self._initial_state()
            m = self._initial_state()
            return [(c[i], m[i]) for i in range(len(c))]

        elif self.rnn_cell_type in ["GRU", "QRNN"]:
            return self._initial_state()
        else:
            raise NotImplementedError("Function initial_state() not implemented for cell type %s" %
                                      self.rnn_cell_type)

    # Returns state of the shape [dilation, 1, hidden_size]
    def _initial_state(self):
        def _helper(type, shape):
            if type == 'random':
                h = np.array(np.random.normal(0, 1.0, shape), dtype=np.float32)
                return np.clip(h, -1, 1).astype(dtype=np.float32)
            elif type == 'zero':
                return np.zeros(shape, np.float32)
        h = []
        for i in range(self.rnn_num_layers):
            dilation = 2 ** i
            h.append(_helper(self.rnn_initial_state, (dilation, 1, self.rnn_hidden_size)))

        return h

    # Converts PyTorch hidden state representation into something that can be saved
    def export_state(self, states):
        if self.rnn_cell_type == "LSTM":
            states_0, states_1 = states[0], states[1]
            return self._export_state(states_0), self._export_state(states_1)

        elif self.rnn_cell_type in ["GRU"]:
            return self._export_state(states)
        else:
            raise NotImplementedError

    def _export_state(self, states):
        states = [torch.transpose(s, 1, 0) for s in states]
        batch_size = states[0].size(0)

        # Placeholder for samples
        samples = [[] for _ in range(batch_size)]
        for i_layer in range(self.rnn_num_layers):
            dilation = 2**i_layer
            layer_state = states[i_layer]
            layer_state = torch.split(layer_state, dilation)

            for s, st in zip(samples, layer_state):
                s.append(st.cpu().data.numpy())

        return samples

    # Convert something that was saved into PyTorch representation
    def import_state(self, states):
        if self.rnn_cell_type == "LSTM":
            states_0, states_1 = [s[:][:][0] for s in states], [s[:][:][1] for s in states]
            return self._import_state(states_0), self._import_state(states_1)

        elif self.rnn_cell_type in ["GRU"]:
            return self._import_state(states)
        else:
            raise NotImplementedError

    # States come from different examples, merge them into one single minibatch
    def _import_state(self, states):
        hidden = []
        for i_layer in range(self.rnn_num_layers):
            s = [sample[i_layer] for sample in states]
            s = np.concatenate(s)
            s = np.swapaxes(s, 1, 0)
            h = Variable(torch.from_numpy(s), requires_grad=False)
            hidden.append(h)
        return hidden

    def offset_size(self, sequence_size):
        return 0


if __name__ == '__main__':
    model = DilatedRnnModel(input_size=2,
                            output_size=2,
                            context_size=0,
                            batch_norm=0,
                            skip_mode='none',
                            rnn_initial_state='random',
                            rnn_hidden_size=3,
                            rnn_num_layers=2,
                            dropout_f=0.0,
                            dropout_h=0.0,
                            dropout_i=0.0,
                            rnn_cell_type='GRU',
                            use_context=0,
                            lasso_selection=0.0)

    initial_states = [model.initial_state() for i in range(2)]
    print('Initial states')
    print(initial_states)

    imported_states = model.import_state(initial_states)
    # print('Imported states')
    # print(imported_states)

    exported_states = model.export_state(imported_states)

    print('Exported states')
    print(exported_states)
    #
    # batch = np.array([[[1, 1], [2, 2], [3, 3], [4, 4]],
    #                   [[11, 11], [12, 12], [13, 13], [14, 14]]]).astype(np.float32)
    #
    # input = Variable(torch.from_numpy(batch))
    # #
    # print('Input')
    # print(input)
    # output, hidden = model.forward(input, imported_states, None)
    # print('Output')
    # print(output)
    # # print('Done')
