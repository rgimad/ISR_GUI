import numpy as np
import cv2

from PyQt5 import QtGui
from PyQt5.QtGui import QPixmap, QImage, QCursor, QPainter, QPen
from PyQt5.QtCore import Qt, QMimeData, QRect
from PyQt5.QtWidgets import QApplication, QLabel, QAction, QMenu, QSizePolicy, QFileDialog

class ImageLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.showContextMenu)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setAlignment(Qt.AlignCenter)
        self.image_name = "image"

    def showContextMenu(self, pos):
        menu = QMenu()
        copyAction = QAction("Копировать", self)
        copyAction.triggered.connect(self.copyImage)
        saveAction = QAction("Сохранить", self)
        saveAction.triggered.connect(self.saveImage)
        menu.addAction(copyAction)
        menu.addAction(saveAction)
        menu.exec_(QCursor.pos())

    def copyImage(self):
        if self.pixmap():
            mimeData = QMimeData()
            mimeData.setImageData(self.pixmap().toImage())
            QApplication.clipboard().setMimeData(mimeData)

    def saveImage(self):
        if self.pixmap():
            filename, _ = QFileDialog.getSaveFileName(self, "Сохранить изображение", self.image_name, "PNG (*.png);;JPG (*.jpg *.jpeg);;BMP (*.bmp)")
            if filename != "":
                self.pixmap().save(filename)

    def setImageName(self, image_name):
        self.image_name = image_name

    def setPixmapFromGrayscaleNumpy(self, img):
        qImg = QImage(img.data, img.shape[1], img.shape[0], img.shape[1], QImage.Format_Grayscale8)
        pixmap = QPixmap(qImg)
        self.setPixmap(pixmap)

################

class RoiImageLabel(ImageLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.roi_choose_mode = False
        self.in_image_x = 0
        self.in_image_y = 0

    def setROICallback(self, roi_callback):
        self.roi_callback = roi_callback

    def enableROIChoose(self):
        self.roi_choose_mode = True

    def disableROIChoose(self):
        self.roi_choose_mode = False

    def loadImage(self, image_path, scaled_maxwidth, scaled_maxheight, roi_real_size):
        self.orig_pixmap = QPixmap.fromImage(QPixmap.toImage(QPixmap(image_path)).convertToFormat(QtGui.QImage.Format_Grayscale8))
        self.orig_numpy = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        self.scaled_pixmap = self.orig_pixmap.scaled(scaled_maxwidth, scaled_maxheight, Qt.AspectRatioMode.KeepAspectRatio)
        self.scaled_pixmap_x = (scaled_maxwidth - self.scaled_pixmap.width()) // 2
        self.scaled_pixmap_y = (scaled_maxheight - self.scaled_pixmap.height()) // 2
        self.roi_real_size = roi_real_size
        self.roi_scaled_size = roi_real_size * self.scaled_pixmap.width() // self.orig_pixmap.width()
        self.setPixmap(self.scaled_pixmap)
        self.setToolTip("Файл: " + image_path + f"\nРазрешение: {self.orig_numpy.shape[1]}x{self.orig_numpy.shape[0]}")


    def mouseMoveEvent(self, event):
        if not self.roi_choose_mode:
            return

        self.in_image_x = event.x() - self.scaled_pixmap_x - self.roi_scaled_size // 2
        self.in_image_y = event.y() - self.scaled_pixmap_y - self.roi_scaled_size // 2
        self.in_image_x = max(self.in_image_x, 0)
        self.in_image_y = max(self.in_image_y, 0)
        self.in_image_x = min(self.in_image_x, self.scaled_pixmap.width() - self.roi_scaled_size)
        self.in_image_y = min(self.in_image_y, self.scaled_pixmap.height() - self.roi_scaled_size)

        img = QPixmap.toImage(self.scaled_pixmap)
        painter = QPainter(img)
        pen = QPen()
        pen.setWidth(2)
        pen.setColor(Qt.red)
        painter.setPen(pen)
        painter.drawRect(self.in_image_x, self.in_image_y, self.roi_scaled_size, self.roi_scaled_size)
        painter.end()
        self.setPixmap(QPixmap.fromImage(img))

    def mouseReleaseEvent(self, event):
        if not self.roi_choose_mode:
            return
        real_x = self.in_image_x * self.orig_pixmap.width() // self.scaled_pixmap.width()
        real_y = self.in_image_y * self.orig_pixmap.height() // self.scaled_pixmap.height()
        # r = QRect(real_x, real_y, self.roi_real_size, self.roi_real_size)
        # img1 = self.orig_pixmap.toImage()
        # orig_roi_pixmap = QPixmap.fromImage(img1.copy(r))
        # # self.setPixmap(orig_roi_pixmap)
        # self.disableROIChoose()
        # self.roi_callback(orig_roi_pixmap)
        orig_roi_numpy = self.orig_numpy[real_y:real_y + self.roi_real_size, real_x:real_x + self.roi_real_size].copy()
        self.disableROIChoose()
        self.roi_callback(orig_roi_numpy)






        