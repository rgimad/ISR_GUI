import numpy as np

from PyQt5.QtGui import QPixmap, QImage, QCursor
from PyQt5.QtCore import Qt, QMimeData
from PyQt5.QtWidgets import QApplication, QLabel, QAction, QMenu, QSizePolicy

class ImageLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.showContextMenu)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setAlignment(Qt.AlignCenter)

    def showContextMenu(self, pos):
        menu = QMenu()
        copyAction = QAction("Копировать", self)
        copyAction.triggered.connect(self.copyImage)
        menu.addAction(copyAction)
        menu.exec_(QCursor.pos())

    def copyImage(self):
        if self.pixmap():
            mimeData = QMimeData()
            mimeData.setImageData(self.pixmap().toImage())
            QApplication.clipboard().setMimeData(mimeData)

    def setPixmapFromGrayscaleNormNumpy(self):
        output_img = (output_img * 255.0).astype(np.uint8) # Clip and convert to uint8
        height, width = output_img.shape
        bytesPerLine = width
        qImg = QImage(output_img.data, width, height, bytesPerLine, QImage.Format_Grayscale8)
        pixmap = QPixmap(qImg)
        self.setPixmap(pixmap)

        