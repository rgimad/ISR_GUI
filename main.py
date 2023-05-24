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

from utils import round_to_multiple

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

        self.cbResearchChooseModel.currentTextChanged.connect(self.research_model_changed)

        self.input_image_filename = None
        
        self.sr_models = dict()

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

        self.labelResearchInputImage.loadImage(self.input_image_filename, self.saResearchInputImage.width() - 5, self.saResearchInputImage.height() - 5, round_to_multiple(self.saResearchGTImage.width() - 10, 4))
        self.saResearchInputImage.setWidget(self.labelResearchInputImage)

    
    def model_inference(self, model, img):
        input_img = img.astype(np.float32) / 255.
        input_img = np.expand_dims(input_img, axis=0) # Add a batch dimension
        input_img = np.expand_dims(input_img, axis=0) # Add a batch dimension
        input_tensor = torch.from_numpy(input_img).to(self.torch_device)

        with torch.no_grad():
            output_tensor = model(input_tensor)

        output_img = output_tensor.cpu().detach().numpy()[0][0] # why second [0]
        output_img = (output_img * 255.0).clip(0, 255).astype(np.uint8) # Clip and convert to uint8
        return output_img


    def roi_chosen_callback(obj, roi_gt):
        global main_window

        main_window.roi_gt = roi_gt.copy()
        
        main_window.labelResearchGTImage.setPixmap(QPixmap(QImage(roi_gt.data, roi_gt.shape[1], roi_gt.shape[0], roi_gt.shape[1], QImage.Format_Grayscale8)))
        main_window.saResearchGTImage.setWidget(main_window.labelResearchGTImage)

        cur_scale = main_window.research_get_current_scale()
        roi_lr = cv2.resize(roi_gt, (roi_gt.shape[1] // cur_scale, roi_gt.shape[0] // cur_scale), interpolation = cv2.INTER_CUBIC)
        roi_bicubic = cv2.resize(roi_lr, (roi_gt.shape[1], roi_gt.shape[0]), interpolation = cv2.INTER_CUBIC)

        main_window.labelResearchBicubicImage.setPixmap(QPixmap(QImage(roi_bicubic.data, roi_gt.shape[1], roi_gt.shape[0], roi_gt.shape[1], QImage.Format_Grayscale8)))
        main_window.saResearchBicubicImage.setWidget(main_window.labelResearchBicubicImage)

        # Calculate metrics for Bicubic
        psnr_bicubic = psnr(roi_gt, roi_bicubic)
        ssim_bicubic = ssim(roi_gt, roi_bicubic)
        # print(f"psnr_bicub = {psnr_bicubic}, ssim_bicub = {ssim_bicubic}")
        main_window.lbResearchBicubicMetrics.setText(f"PSNR: {psnr_bicubic:.2f} dB\nSSIM: {ssim_bicubic:.4f}")

        QApplication.setOverrideCursor(Qt.WaitCursor)

        model_name, model_fname = main_window.research_get_current_model_name()

        # Load model and weight if not loaded yet
        if model_fname not in main_window.sr_models:
            if model_name == "FSRCNN":
                main_window.sr_models[model_fname] = FSRCNN(cur_scale)
            elif model_name == "EDSR":
                main_window.sr_models[model_fname] = EDSR(cur_scale)
            elif model_name == "VDSR":
                main_window.sr_models[model_fname] = VDSR()
            else:
                print("Unknown model")

            main_window.sr_models[model_fname].load_state_dict(torch.load(model_fname, map_location=('cpu' if main_window.torch_device.type != 'cuda' else None))['state_dict'])
            main_window.sr_models[model_fname].to(main_window.torch_device)
            main_window.sr_models[model_fname].eval()
            print(f"Model {model_fname} loaded and ready")
        else:
            pass

        # Inference
        if model_name == "VDSR": # For pre-upsampling methods we pass bicubic interpolated to the model
            roi_sr = main_window.model_inference(main_window.sr_models[model_fname], roi_bicubic)
        else: # For post upsampling methods we pass LR to the model
            roi_sr = main_window.model_inference(main_window.sr_models[model_fname], roi_lr)

        main_window.labelResearchSRImage.setPixmap(QPixmap(QImage(roi_sr.data, roi_sr.shape[1], roi_sr.shape[0], roi_sr.shape[1], QImage.Format_Grayscale8)))
        main_window.saResearchSRImage.setWidget(main_window.labelResearchSRImage)

        # Calculate metrics for SR
        # print(roi_gt.shape, roi_sr.shape)
        psnr_sr = psnr(roi_gt, roi_sr)
        ssim_sr = ssim(roi_gt, roi_sr)
        # print(f"psnr_sr = {psnr_sr}, ssim_sr = {ssim_sr}")
        main_window.lbResearchSRMetrics.setText(f"PSNR: {psnr_sr:.2f} dB\nSSIM: {ssim_sr:.4f}")

        QApplication.restoreOverrideCursor()

    
    def research_get_current_model_name(self):
        model_name = self.cbResearchChooseModel.currentText()
        return model_name, model_name.lower() + "_ir_" + ("x2" if self.rbResearch_x2.isChecked() else "x4") + ".pth.tar"

    
    def research_get_current_scale(self):
        return 2 if self.rbResearch_x2.isChecked() else 4
    
    def research_model_changed(self):
        self.roi_chosen_callback(self.roi_gt)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())

