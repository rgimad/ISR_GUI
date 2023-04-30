from PyQt5 import QtWidgets, QtGui, uic
from PyQt5.QtGui import QPixmap, QPalette
from PyQt5.QtWidgets import QApplication, QMainWindow, QScrollArea, QFileDialog, QLabel
import sys

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        uic.loadUi('isr.ui', self)
        self.labelInputImage = QLabel()
        self.labelOutputImage = QLabel()

        self.btnChooseImage.clicked.connect(self.choose_input_image)

    def choose_input_image(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.bmp)")
        if filename:
            pixmap = QPixmap(filename)
            self.labelInputImage.setPixmap(pixmap)
            self.scrollAreaInputImage.setWidget(self.labelInputImage)
            self.labelOutputImage.clear()

            # self.labelOutputImage.setPixmap(pixmap)
            # self.scrollAreaOutputImage.setWidget(self.labelOutputImage)



if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())

