import sys 
from PyQt5.QtGui import *
from PyQt5.QtWidgets import * 
from PyQt5.QtCore import * 
import cv2 as cv 

class MainWindow(QWidget):
    def __init__(self):
        super(MainWindow,self).__init__()

        self.VBL = QVBoxLayout()

        self.FeedLabel = QLabel()
        self.VBL.addWidget(self.FeedLabel)

        self.cancelBTN = QPushButton("Cancel")
        self.cancelBTN.clicked.connect(self.CancelFeed)
        self.VBL.addWidget(self.cancelBTN)

        self.Worker1 = Worker1()

        self.Worker1.start()
        self.Worker1.ImageUpdate.connect(self.ImageUpdateSlot)
        self.setLayout(self.VBL)

    def ImageUpdateSlot(self, Image):
        self.FeedLabel.setPixmap(QPixmap.fromImage(Image))

    def CancelFeed(self):
        self.Worker1.stop()

class Worker1(QThread):
    ImageUpdate = pyqtSignal(QImage)
    def run(self):
        self.ThreadActive = True
        Capture = cv.VideoCapture(0)
        while self.ThreadActive:
            ret, frame = Capture.read()
            if ret:
                Image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                FlippedImage = cv.flip(Image,1)
                ConverToQtFormat = QImage(FlippedImage.data, FlippedImage.shape[1], FlippedImage.shape[0], QImage.Format_RGB888)   
                Pic = ConverToQtFormat.scaled(640,480, Qt.KeepAspectRatio)
                self.ImageUpdate.emit(Pic)
        def stop(self):
            self.ThreadActive = False
            self.quit()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    root = MainWindow()
    root.show()
    sys.exit(app.exec())
