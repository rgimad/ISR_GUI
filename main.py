import os
import sys

import cv2
import numpy as np
import math

from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim, mean_squared_error

from PyQt5 import QtWidgets, QtGui, uic
from PyQt5.QtGui import QPixmap, QPalette, QImage, QCursor, QPainter, QPen
from PyQt5.QtCore import Qt, QMimeData
from PyQt5.QtWidgets import QApplication, QMainWindow, QScrollArea, QFileDialog, QLabel, QMessageBox, QAction, QMenu, QSizePolicy

import torch
from image_label import ImageLabel, RoiImageLabel
from fsrcnn_ir_model import FSRCNN
from vdsr_ir_model import VDSR
from edsr_ir_model import EDSR

main_window = None

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        uic.loadUi('isr.ui', self)

        global main_window
        main_window = self

        self.labelResearchInputImage = RoiImageLabel()
        self.labelResearchInputImage.setROICallback(self.roi_chosen_callback)
        self.labelResearchBicubicImage = ImageLabel()
        self.labelResearchSRImage = ImageLabel()
        self.labelResearchGTImage = ImageLabel()

        self.btnResearchChooseImage.clicked.connect(self.research_process_input_image)
        # self.btnSaveResult.clicked.connect(self.save_output_image)
        # self.btnDoSR.clicked.connect(self.do_super_resolution)

        self.btnResearchChooseROI.clicked.connect(self.research_choose_roi)

        self.input_image_filename = None
        self.fsrcnn_x2_model = None
        self.vdsr_x2_model = None
        self.edsr_x2_model = None

        self.torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"torch device is {self.torch_device}")

    def research_choose_roi(self):
        if self.labelResearchInputImage.pixmap() != None:
            self.labelResearchInputImage.enableROIChoose()

    def research_process_input_image(self):
        # print(self.get_current_model_name())

        self.input_image_filename, _ = QFileDialog.getOpenFileName(self, "Выбрать изображение", "", "Изображения (*.png *.jpg *.bmp)")
        if self.input_image_filename == "":
            return

        self.labelResearchInputImage.loadImage(self.input_image_filename, self.saResearchInputImage.width() - 5, self.saResearchInputImage.height() - 5, self.saResearchGTImage.width() - 5)
        self.saResearchInputImage.setWidget(self.labelResearchInputImage)


    def roi_chosen_callback(obj, roi):
        # print(f"callback1({x})")
        global main_window
        # main_window.labelResearchGTImage.setPixmap(roi_pixmap)
        
        main_window.labelResearchGTImage.setPixmap(QPixmap(QImage(roi.data, roi.shape[1], roi.shape[0], roi.shape[1], QImage.Format_Grayscale8)))
        main_window.saResearchGTImage.setWidget(main_window.labelResearchGTImage)

        cur_scale = main_window.research_get_current_scale()
        roi_downscaled = cv2.resize(roi, (roi.shape[1] // cur_scale, roi.shape[0] // cur_scale), interpolation = cv2.INTER_CUBIC)
        roi_restored_bicubic = cv2.resize(roi_downscaled, (roi.shape[1], roi.shape[0]), interpolation = cv2.INTER_CUBIC)

        main_window.labelResearchBicubicImage.setPixmap(QPixmap(QImage(roi_restored_bicubic.data, roi.shape[1], roi.shape[0], roi.shape[1], QImage.Format_Grayscale8)))
        main_window.saResearchBicubicImage.setWidget(main_window.labelResearchBicubicImage)

        psnr_bicubic = psnr(roi, roi_restored_bicubic)
        ssim_bicubic = ssim(roi, roi_restored_bicubic)
        # print(f"psnr_bicub = {psnr_bicubic}, ssim_bicub = {ssim_bicubic}")
        main_window.lbResearchBicubicMetrics.setText(f"PSNR: {psnr_bicubic:.2f} dB\nSSIM: {ssim_bicubic:.4f}")

        # err = mean_squared_error(roi, roi_restored_bicubic)
        # psnr2 = 10 * np.log10((255 ** 2) / err)
        # print(psnr2)
        

    def research_get_current_model_name(self):
        return self.cbResearchChooseModel.currentText().lower() + "_ir_" + ("x2" if self.rbResearch_x2.isChecked() else "x4") + ".pth.tar"

    
    def research_get_current_scale(self):
        return 2 if self.rbResearch_x2.isChecked() else 4


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())

