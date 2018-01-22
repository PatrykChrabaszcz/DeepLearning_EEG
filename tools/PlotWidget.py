from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPen, QColor, QBrush
from pyqtgraph import GraphicsLayoutWidget, PlotCurveItem, PlotWidget, ViewBox, AxisItem, PlotItem
from PyQt5.QtCore import Qt
import numpy as np


class ChartWidget(GraphicsLayoutWidget):
    class Zoomer:
        def __init__(self, x_scale=1000.0, y_scale=1.0, x_zoom=0.01, y_zoom=0.01):
            self.x_offset = 0
            self.x_scale = x_scale
            self.y_scale = y_scale
            self.x_zoom = x_zoom
            self.y_zoom = y_zoom

            self.curr_x = 0
            self.curr_y = 0

        def set_position(self, x, y):
            self.curr_x = x
            self.curr_y = y

        def left_button_moved(self, x, y, width):
            dx = x - self.curr_x
            dx = -dx * self.x_scale / width

            self.x_offset = max(0, dx + self.x_offset)

            self.curr_x = x
            self.curr_y = y
            return self.x_offset, self.x_offset + self.x_scale

        def right_button_moved(self, x, y):
            dy = (y - self.curr_y) * self.y_zoom
            self.y_scale *= np.exp(dy)
            y_range = (-self.y_scale, self.y_scale)

            dx = (x - self.curr_x) * self.x_zoom

            self.x_scale = max(10, self.x_scale * np.exp(dx))
            x_range = (self.x_offset, self.x_offset + self.x_scale)

            self.curr_x = x
            self.curr_y = y

            return x_range, y_range

    def __init__(self, parent=None):
        super().__init__(parent)

        self.zoomer = ChartWidget.Zoomer()
        self.vb = None
        self.setup()

    def setup(self):
        self.ci.setContentsMargins(0, 0, 0, 0)
        self.ci.setSpacing(0)

    def plot(self, data, predictions):
        self.vb = None
        self.clear()

        channels = data.shape[0]
        lenght = data.shape[1]

        plots = []
        for i, d in enumerate(data):
            p = self.addPlot(y=d, pen=['r', 'g', 'b'][i%3], antialias=True)
            p.hideAxis('left')
            p.hideAxis('bottom')
            self.nextRow()
            vb = p.getViewBox()
            vb.setFlag(vb.ItemClipsChildrenToShape, False)
            x_range, y_range = self.zoomer.right_button_moved(0.0, 0.0)
            vb.setXRange(*x_range)
            vb.setYRange(*y_range)
            if self.vb is not None:
                vb.setXLink(self.vb)
                vb.setYLink(self.vb)
            else:
                self.vb = vb

            plots.append(p)

        p = self.addPlot(y=predictions, size=1, rows=30)
        p.showGrid(x=True, y=True)
        vb = p.getViewBox()
        x_range, y_range = self.zoomer.right_button_moved(0.0, 0.0)
        vb.setXRange(*x_range)
        vb.setYRange(-0.2, 1.2)
        vb.setXLink(self.vb)

    def mousePressEvent(self, ev):
        self.zoomer.set_position(ev.x(), ev.y())

    def mouseMoveEvent(self, ev):
        if Qt.RightButton == ev.buttons():
            x_range, y_range = self.zoomer.right_button_moved(ev.x(), ev.y())
            self.vb.setXRange(*x_range)
            self.vb.setYRange(*y_range)
        if Qt.LeftButton == ev.buttons():
            self.vb.setXRange(*self.zoomer.left_button_moved(ev.x(), ev.y(), self.width()))

    def mouseReleaseEvent(self, ev):
        pass
