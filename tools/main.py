from PyQt5.QtWidgets import QApplication
from tools.MainWindow import MainWindow
import sys


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(500, 500)
    window.show()

    return app.exec()

if __name__ == '__main__':
    main()
