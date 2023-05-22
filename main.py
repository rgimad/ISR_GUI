import os
import sys

import cv2
import numpy as np
import math

from PyQt5 import QtWidgets, QtGui, uic
from PyQt5.QtGui import QPixmap, QPalette, QImage, QCursor, QPainter, QPen
from PyQt5.QtCore import Qt, QMimeData
from PyQt5.QtWidgets import QApplication, QMainWindow, QScrollArea, QFileDialog, QLabel, QMessageBox, QAction, QMenu, QSizePolicy

import torch
from image_label import ImageLabel
from fsrcnn_ir_model import FSRCNN
from vdsr_ir_model import VDSR
from edsr_ir_model import EDSR


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        uic.loadUi('isr.ui', self)
        self.labelInputImage = ImageLabel()
        self.labelBicubicImage = ImageLabel()
        self.labelOutputImage = ImageLabel()

        self.btnResearchChooseImage.clicked.connect(self.reserach_process_input_image)
        # self.btnSaveResult.clicked.connect(self.save_output_image)
        # self.btnDoSR.clicked.connect(self.do_super_resolution)

        self.input_image_filename = None
        self.fsrcnn_x2_model = None
        self.vdsr_x2_model = None
        self.edsr_x2_model = None

        self.torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"torch device is {self.torch_device}")

    def reserach_process_input_image(self):
        # print(self.get_current_model_name())

        self.input_image_filename, _ = QFileDialog.getOpenFileName(self, "Выбрать изображение", "", "Изображения (*.png *.jpg *.bmp)")
        # QMessageBox.about(self, "", f"'{self.input_image_filename}'")
        if self.input_image_filename == "":
            return
        
        pixmap = QPixmap.fromImage(QPixmap.toImage(QPixmap(self.input_image_filename)).convertToFormat(QtGui.QImage.Format_Grayscale8))
        self.scaledInputImagePixmap = pixmap.scaled(self.saResearchInputImage.width() - 5, self.saResearchInputImage.height() - 5, Qt.AspectRatioMode.KeepAspectRatio)
        self.labelInputImage.setPixmap(self.scaledInputImagePixmap)
        self.saResearchInputImage.setWidget(self.labelInputImage)

        img = QPixmap.toImage(self.labelInputImage.pixmap())
        painter = QPainter(img)
        pen = QPen()
        pen.setWidth(4)
        pen.setColor(Qt.red)
        painter.setPen(pen)
        # painter.drawPoint(55,55)
        roi_screen_width = self.saResearchGTImage.width() * self.scaledInputImagePixmap.width() // pixmap.width()
        roi_screen_height = self.saResearchGTImage.height() * self.scaledInputImagePixmap.height() // pixmap.height()
        painter.drawRect(55, 55, roi_screen_width, roi_screen_height)
        painter.end()
        self.labelInputImage.setPixmap(QPixmap.fromImage(img))

        # if filename:
        #     # print(filename)
        #     self.input_image_filename = filename
        #     # pixmap = QPixmap(filename)
        #     pixmap = QPixmap.fromImage(QPixmap.toImage(QPixmap(filename)).convertToFormat(QtGui.QImage.Format_Grayscale8))
        #     self.labelInputImage.setPixmap(pixmap)
        #     self.scrollAreaInputImage.setWidget(self.labelInputImage)
        #     self.scrollAreaInputImage.setToolTip(self.input_image_filename)
        #     # self.labelOutputImage.clear() #!!

        #     # tmp, for debug:
        #     # self.labelOutputImage.setPixmap(pixmap)
        #     # self.scrollAreaOutputImage.setWidget(self.labelOutputImage)


    def get_current_model_name(self):
        return self.cbResearchChooseModel.currentText().lower() + "_ir_" + ("x2" if self.rbResearch_x2.isChecked() else "x4") + ".pth.tar"

    def save_output_image(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Save Image", os.path.basename(os.path.splitext(self.input_image_filename)[0]) + "_SR_" + self.cbChooseModel.currentText(), "Image Files (*.png *.jpg *.bmp)")
        out_pixmap = self.labelOutputImage.pixmap()
        if out_pixmap is not None and filename:
            out_pixmap.save(filename)

    def do_bicubic(self):
        # do bicubic interpolation:
        # assume label1 is the input QLabel and label2 is the output QLabel
        pixmap1 = self.labelInputImage.pixmap()
        img1 = pixmap1.toImage()
        # upsample img1 using bicubic interpolation
        img2 = QImage(img1.width() * 2, img1.height() * 2, QImage.Format_Grayscale8)
        scaled_img = img1.scaled(img2.width(), img2.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        img2 = scaled_img.convertToFormat(QImage.Format_Grayscale8)
        # create a QPixmap from img2 and set it to label2
        pixmap2 = QPixmap.fromImage(img2)
        self.labelBicubicImage.setPixmap(pixmap2)
        self.scrollAreaBicubicImage.setWidget(self.labelBicubicImage)

    def model_do_inference(self, image, model):
        orig_image = image
        # image = np.transpose(image, (2, 0, 1)) # Transpose the image to (channels, height, width)
        image = np.expand_dims(image, axis=0) # Add a batch dimension
        image = np.expand_dims(image, axis=0) # Add a batch dimension #
        input_tensor = torch.from_numpy(image).to(self.torch_device) # Convert to a PyTorch tensor and move to the device

        with torch.no_grad():
            output_tensor = model(input_tensor)

        output_img = output_tensor.cpu().detach().numpy()[0][0] # why second [0]
        # output_img = np.transpose(output_img, (1, 2, 0))
        # output_img = (output_img * 255.0).clip(0, 255).astype(np.uint8) # Clip and convert to uint8

        psnr = 10. * math.log10(1. / np.square(np.subtract(orig_image, output_img)).mean())
        print(f"psnr = {psnr}")

        output_img = (output_img * 255.0).astype(np.uint8) # Clip and convert to uint8
        # print(output_img)
        # cv2.imwrite('fdfh.png', output_img)

        # convert numpy image to QImage
        height, width = output_img.shape
        bytesPerLine = width
        qImg = QImage(output_img.data, width, height, bytesPerLine, QImage.Format_Grayscale8)
        pixmap = QPixmap(qImg)
        self.labelOutputImage.setPixmap(pixmap)
        self.scrollAreaOutputImage.setWidget(self.labelOutputImage) 

    def do_super_resolution(self):
        model_name = self.cbChooseModel.currentText()
        if self.input_image_filename == None:
            QMessageBox.about(self, "Ошибка", "Сначала выберите исходное изображение")
            return
        
        QApplication.setOverrideCursor(Qt.WaitCursor)

        self.do_bicubic()

        if model_name == "FSRCNN_x2":
            if self.fsrcnn_x2_model == None:
                self.fsrcnn_x2_model = FSRCNN(2)
                self.fsrcnn_x2_model.load_state_dict(torch.load('fsrcnn_ir_x2.pth.tar', map_location=('cpu' if self.torch_device.type != 'cuda' else None))['state_dict'])
                self.fsrcnn_x2_model.to(self.torch_device)
                self.fsrcnn_x2_model.eval()
                print(f"model {model_name} loaded and ready")

            input_img = cv2.imread(self.input_image_filename, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.
            self.model_do_inference(input_img, self.fsrcnn_x2_model)

        elif model_name == "VDSR_x2":
            if self.vdsr_x2_model == None:
                self.vdsr_x2_model = VDSR()
                self.vdsr_x2_model.load_state_dict(torch.load('vdsr_ir_x2.pth.tar', map_location=('cpu' if self.torch_device.type != 'cuda' else None))['state_dict'])
                self.vdsr_x2_model.to(self.torch_device)
                self.vdsr_x2_model.eval()
                print(f"model {model_name} loaded and ready")

            input_img = cv2.imread(self.input_image_filename, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.
            bicubic_input_img = cv2.resize(input_img, (input_img.shape[1]*2, input_img.shape[0]*2), interpolation = cv2.INTER_CUBIC)
            self.model_do_inference(bicubic_input_img, self.vdsr_x2_model)

        elif model_name == "EDSR_x2":
            if self.edsr_x2_model == None:
                self.edsr_x2_model = EDSR(2)
                self.edsr_x2_model.load_state_dict(torch.load('edsr_ir_x2.pth.tar', map_location=('cpu' if self.torch_device.type != 'cuda' else None))['state_dict'])
                self.edsr_x2_model.to(self.torch_device)
                self.edsr_x2_model.eval()
                print(f"model {model_name} loaded and ready")

            input_img = cv2.imread(self.input_image_filename, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.
            self.model_do_inference(input_img, self.edsr_x2_model)

        else:
            QMessageBox.about(self, "error", "not implemented yet")

        QApplication.restoreOverrideCursor()





if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())

