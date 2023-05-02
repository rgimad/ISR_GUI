import os
import sys
from PyQt5 import QtWidgets, QtGui, uic
from PyQt5.QtGui import QPixmap, QPalette
from PyQt5.QtWidgets import QApplication, QMainWindow, QScrollArea, QFileDialog, QLabel, QMessageBox

import cv2
import torch

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        uic.loadUi('isr.ui', self)
        self.labelInputImage = QLabel()
        self.labelOutputImage = QLabel()

        self.btnChooseImage.clicked.connect(self.choose_input_image)
        self.btnSaveResult.clicked.connect(self.save_output_image)
        self.btnDoSR.clicked.connect(self.do_super_resolution)

        self.input_image_filename = None
        self.fsrcnn_model = None

    def choose_input_image(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.bmp)")
        if filename:
            # print(filename)
            self.input_image_filename = filename
            pixmap = QPixmap(filename)
            self.labelInputImage.setPixmap(pixmap)
            self.scrollAreaInputImage.setWidget(self.labelInputImage)
            self.labelOutputImage.clear()

            # tmp, for debug:
            self.labelOutputImage.setPixmap(pixmap)
            self.scrollAreaOutputImage.setWidget(self.labelOutputImage)

    def save_output_image(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Save Image", os.path.basename(os.path.splitext(self.input_image_filename)[0]) + "_SR_" + self.cbChooseModel.currentText(), "Image Files (*.png *.jpg *.bmp)")
        out_pixmap = self.labelOutputImage.pixmap()
        if out_pixmap is not None and filename:
            out_pixmap.save(filename)

    def do_super_resolution(self):
        model_name = self.cbChooseModel.currentText()
        if model_name == "FSRCNN":
            if self.fsrcnn_model == None:
                self.fsrcnn_model = torch.load('fsrcnn_x2-T91-f791f07f.pth.tar')
                # with open('fhjf.txt', 'w') as f:
                #     f.write(str(self.fsrcnn_model))
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                # self.fsrcnn_model.to(device) # this won't work cuz in this case its just state_dist, so i nedd firtly provide model impl, at then load separately state dict for it
        else:
            QMessageBox.about(self, "error", "not implemented yet")



if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())

