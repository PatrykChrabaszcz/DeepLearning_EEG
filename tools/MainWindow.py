import numpy as np
import torch
from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QTextEdit, QWidget, QAction, QFileDialog
from torch.autograd import Variable

from src.data_reading.data_reader import AnomalyDataReader
from src.dl_pytorch.model import SimpleRNN
from tools.PlotWidget import ChartWidget


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setCentralWidget(QWidget(self))

        self.main_layout = QVBoxLayout(self.centralWidget())

        self.chartWidget = ChartWidget()

        #self.chartWidget.plot(np.random.normal(size=[5, 1000]), np.random.choice([1, 0],1000))
        self.main_layout.addWidget(self.chartWidget)
        self.main_layout.addWidget(QTextEdit())

        # self.timer = QTimer()
        # self.timer.timeout.connect(self.whaat)
        # self.timer.start(1000)

        self.file_menu = self.menuBar().addMenu('File')
        self.open_file_action = QAction('Open File')
        self.open_file_action.triggered.connect(self.read_file)
        self.file_menu.addAction(self.open_file_action)

        self.load_network_action = QAction('Load Network')
        self.load_network_action.triggered.connect(self.load_network)
        self.file_menu.addAction(self.load_network_action)

        self.model = SimpleRNN(22, 128, 3, 2, 'GRU')

    def read_file(self):
        file_name = QFileDialog.getOpenFileName(directory='/mhome/chrabasp/data_full/train')[0]
        example = AnomalyDataReader.ExampleInfo((1, 2, 5), file_name, 0)
        example.reset(example.length, randomize=False)

        data, time, label, example_id = example.read_data((0, example.length))

        probabilities = self.run_network(data)
        data = np.transpose(data)
        self.chartWidget.plot(data, probabilities)

    def load_network(self):
        file_name = QFileDialog.getOpenFileName(directory='../models')[0]
        self.model.load_model(file_name)

    def run_network(self, data):
        print(data.shape)
        data = data[:1000, :]
        data = np.expand_dims(data, axis=0)
        print(data.shape)
        hidden = self.model.initial_state()
        hidden = self.model.import_state([hidden])

        # for i, d in enumerate(data):
        #     print(i)
        #     d = np.expand_dims(np.expand_dims(d, axis=0), axis=0)
        #
        #     print(d.shape)

        d = Variable(torch.from_numpy(data))
        output, hidden = self.model.forward(d, hidden)

            # h = hidden.data.numpy()
            # print('H_max %g' % h.max())
            # print('H_min %g' % h.min())
            # print('H_norm %g' % np.sum(np.square(h)))
        predictions = output.data.numpy()[0]
        predictions_exp = np.exp(predictions)
        probabilities = predictions_exp / np.sum(predictions_exp, axis=1, keepdims=True)
        print(probabilities)
        return probabilities[:, 1].flatten()
        # data = Variable(torch.from_numpy(data))
        #
        # output, hidden = self.model.forward(data, hidden)
        # print(output)

