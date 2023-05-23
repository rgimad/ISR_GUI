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
        self.labelResearchInputImage.enableROIChoose()

    def research_process_input_image(self):
        # print(self.get_current_model_name())

        self.input_image_filename, _ = QFileDialog.getOpenFileName(self, "Выбрать изображение", "", "Изображения (*.png *.jpg *.bmp)")
        if self.input_image_filename == "":
            return

        self.labelResearchInputImage.loadImage(self.input_image_filename, self.saResearchInputImage.width() - 5, self.saResearchInputImage.height() - 5, self.saResearchGTImage.width() - 5)
        self.saResearchInputImage.setWidget(self.labelResearchInputImage)


    def roi_chosen_callback(obj, roi_pixmap):
        # print(f"callback1({x})")
        global main_window
        main_window.labelResearchGTImage.setPixmap(roi_pixmap)
        main_window.saResearchGTImage.setWidget(main_window.labelResearchGTImage)

    def get_current_model_name(self):
        return self.cbResearchChooseModel.currentText().lower() + "_ir_" + ("x2" if self.rbResearch_x2.isChecked() else "x4") + ".pth.tar"

    



if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())

