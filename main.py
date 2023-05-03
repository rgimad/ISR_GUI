import os
import sys
from PyQt5 import QtWidgets, QtGui, uic
from PyQt5.QtGui import QPixmap, QPalette, QImage
from PyQt5.QtWidgets import QApplication, QMainWindow, QScrollArea, QFileDialog, QLabel, QMessageBox

import cv2
import torch
import numpy as np

from fsrcnn_model_test1 import FSRCNN

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

        self.torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"torch device is {self.torch_device}")

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
            # self.labelOutputImage.setPixmap(pixmap)
            # self.scrollAreaOutputImage.setWidget(self.labelOutputImage)

    def save_output_image(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Save Image", os.path.basename(os.path.splitext(self.input_image_filename)[0]) + "_SR_" + self.cbChooseModel.currentText(), "Image Files (*.png *.jpg *.bmp)")
        out_pixmap = self.labelOutputImage.pixmap()
        if out_pixmap is not None and filename:
            out_pixmap.save(filename)

    def do_super_resolution(self):
        model_name = self.cbChooseModel.currentText()
        if self.input_image_filename == None:
            QMessageBox.about(self, "Ошибка", "Сначала выберите исходное изображение")
            return
        if model_name == "FSRCNN":
            if self.fsrcnn_model == None:
                self.fsrcnn_model = FSRCNN(2)
                self.fsrcnn_model.load_state_dict(torch.load('fsrcnn_x2-T91-f791f07f.pth.tar')['state_dict'])
                self.fsrcnn_model.to(self.torch_device)
                self.fsrcnn_model.eval()
                print('model loaded and ready')

            input_img = cv2.imread(self.input_image_filename, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.
            # input_img = np.transpose(input_img, (2, 0, 1)) # Transpose the image to (channels, height, width)
            input_img = np.expand_dims(input_img, axis=0) # Add a batch dimension
            input_img = np.expand_dims(input_img, axis=0) # Add a batch dimension #
            input_tensor = torch.from_numpy(input_img).to(self.torch_device) # Convert to a PyTorch tensor and move to the device

            with torch.no_grad():
                output_tensor = self.fsrcnn_model(input_tensor)

            output_img = output_tensor.cpu().detach().numpy()[0][0] # why second [0]
            # output_img = np.transpose(output_img, (1, 2, 0))
            # output_img = (output_img * 255.0).clip(0, 255).astype(np.uint8) # Clip and convert to uint8
            output_img = (output_img * 255.0).astype(np.uint8) # Clip and convert to uint8
            print(output_img)
            cv2.imwrite('fdfh.png', output_img)

            # convert numpy image to QImage
            height, width = output_img.shape
            bytesPerLine = width
            qImg = QImage(output_img.data, width, height, bytesPerLine, QImage.Format_Grayscale8)
            pixmap = QPixmap(qImg)
            self.labelOutputImage.setPixmap(pixmap)
            self.scrollAreaOutputImage.setWidget(self.labelOutputImage)

                
                
        else:
            QMessageBox.about(self, "error", "not implemented yet")



if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())

