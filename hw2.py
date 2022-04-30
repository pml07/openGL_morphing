import sys
from Ui_morph_interface import Ui_MainWindow
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox, QDockWidget, QListWidget, QLabel, QWidget
from PyQt5.QtGui import *
import cv2
import numpy as np
import math

class mywindow(QtWidgets.QMainWindow, Ui_MainWindow):

    def __init__(self):
        super(mywindow, self).__init__()
        self.setupUi(self)
        self.Morphing.clicked.connect(self.runClicked)
        self.source_points = []
        self.destination_points = []
        self.source_lines = []
        self.destination_lines = []
        self.source_origin = cv2.imread('./images/women.jpg')
        self.source_img = self.source_origin.copy()
        self.destination_origin = cv2.imread('./images/cheetah.jpg')
        self.destination_img = self.destination_origin.copy()
        self.setImage()
        self.Draw.clicked.connect(self.showImage)
        self.Finish.clicked.connect(self.finDraw)
        self.Finish.setVisible(False)
        self.Show.clicked.connect(self.animate)
        self.source_linepair = [[np.array([64, 22]), np.array([96, 32])], [np.array([133,  33]), np.array([172,  27])], [np.array([60, 44]),
         np.array([93, 48])], [np.array([134,  47]), np.array([167,  48])], [np.array([114,  34]), np.array([114, 103])], [np.array([ 89, 130]),
          np.array([137, 130])], [np.array([38, 48]), np.array([ 59, 146])], [np.array([ 64, 153]), np.array([ 71, 186])], [np.array([185,  50]),
           np.array([169, 142])], [np.array([165, 156]), np.array([159, 186])]]
        self.destination_linepair = [[np.array([25,  4]), np.array([72,  5])], [np.array([181,   7]), np.array([233,   4])], [np.array([29, 17]),
         np.array([78, 21])], [np.array([179,  21]), np.array([227,  17])], [np.array([127,   8]), np.array([128, 163])], [np.array([ 82, 180]),
          np.array([175, 180])], [np.array([ 2, 12]), np.array([  3, 139])], [np.array([  4, 146]), np.array([  5, 187])], [np.array([252,  13]),
           np.array([254, 146])], [np.array([253, 156]), np.array([253, 182])]]

    def setImage(self):
        source = QPixmap('./result/Source.jpg')
        destination = QPixmap('./result/Destination.jpg')
        self.sourceimg.setPixmap(source)
        self.destinationimg.setPixmap(destination)

    def showImage(self):
        self.Finish.setVisible(True)
        cv2.namedWindow("Source Image")
        cv2.imshow("Source Image", self.source_img)
        cv2.namedWindow("Destination Image")
        cv2.imshow("Destination Image", self.destination_img)


        def getLine(event, x, y, flags, param):
            winname = param[0]
            img = param[1]
            point_list = param[2]
            line_list = param[3]
            if event == cv2.EVENT_LBUTTONDOWN:
                cv2.circle(img, (x, y), 2, (0, 0, 255),2)
                point_list.append(np.array([x, y]))
                cv2.imshow(winname, img)
                if len(param[2]) % 2 == 0:
                    cv2.line(img, (point_list[-2][0], point_list[-2][1]), (x, y), (0, 0, 255), 2)
                    line = [np.array(point_list[-2]),np.array([x, y])]
                    line_list.append(line)
                    cv2.imshow(winname, img)

        cv2.setMouseCallback("Source Image", getLine, param=["Source Image", self.source_img, self.source_points, self.source_lines])
        cv2.setMouseCallback("Destination Image", getLine, param=["Destination Image", self.destination_img, self.destination_points, self.destination_lines])


    def lineInterpolate(self, source_lines, destination_lines, ratio):
        inter_lines = []
        for source, destination in zip(source_lines, destination_lines):
            start_point = (1 - ratio) * source[0] + ratio * destination[0]
            end_point = (1 - ratio) * source[1] + ratio * destination[1]
            inter_lines.append([start_point,end_point])

        return inter_lines

    def getU(self, x,line):
        P = np.array(line[0])
        Q = np.array(line[1])
        length = np.linalg.norm(Q-P)
        u = np.dot((x-P),(Q-P))/(length*length)
        return u

    def getV(self, x,line):
        P = np.array(line[0])
        Q = np.array(line[1])
        QP = Q-P
        per_QP = np.array([QP[1], -QP[0]])
        length = np.linalg.norm(Q-P)
        v = np.dot((x-P),per_QP)/length
        return v

    def getdesX(self, x,u,v,line):
        P = np.array(line[0])
        Q = np.array(line[1])
        QP = Q-P
        per_QP = np.array([QP[1], -QP[0]])
        length = np.linalg.norm(QP)
        des_x = P + u*(QP)+ (v*per_QP/length)
        return des_x

    def getWeight(self, des_X,line):
        p = int(self.Getp.text())
        a = int(self.Geta.text())
        b = int(self.Getb.text())

        P = line[0]
        Q = line[1]
        length = np.linalg.norm(Q-P)

        u = self.getU(des_X, line)

        if(u > 1.0):
            d = np.linalg.norm(des_X-Q)
        elif(u < 0):
            d = np.linalg.norm(des_X-P)
        else:
            d = abs(self.getV(des_X,line))

        weight = pow((pow(length*length, p)/(a+d)), b)

        return weight

    def bilinear(self, img,x,y):
        h, w, _ = img.shape
        x_floor,y_floor = math.floor(x), math.floor(y)
        x_ceil,y_ceil = math.ceil(x), math.ceil(y)

        if x_ceil >= w:
            x_ceil = w - 1
        if y_ceil >= h:
            y_ceil = h - 1

        a, b = x-x_floor, y-y_floor

        result = (1 - a) * (1 - b) * img[y_floor, x_floor] + a * (1 - b) * img[y_ceil, x_floor] + (1 - a) * b * img[y_floor, x_ceil] + a * b *img[y_ceil, x_ceil]

        return result

    def warping(self, source_img,source_lines,inter_lines):
        h, w, _ = source_img.shape
        warp_img = np.empty_like(source_img)
        for i in range(w):
            for j in range(h):
                XSum = np.array([0,0])
                WeightSum = 0
                for src_line, des_line in zip(source_lines, inter_lines):
                    x = np.array([i,j])
                    u = self.getU(x, des_line)
                    v = self.getV(x, des_line)
                    des_X = self.getdesX(x, u, v, src_line)
                    weight = self.getWeight(des_X, des_line)
                    XSum = XSum + des_X*weight
                    WeightSum += weight
                des_X = XSum/WeightSum

                if des_X[0] < 0:
                    des_X[0] = 0
                elif des_X[0] >= w:
                    des_X[0] = w - 1

                if des_X[1] < 0:
                    des_X[1] = 0
                elif des_X[1] >= h:
                    des_X[1] = h - 1

                warp_img[j,i] = self.bilinear(source_img, des_X[0], des_X[1])
        return warp_img

    def runClicked(self):
        print("calculate")
        if not self.source_lines:
            self.source_lines = self.source_linepair
        if not self.destination_lines:
            self.destination_lines = self.destination_linepair

        t = float(self.GetT.text())
        inter_lines = self.lineInterpolate(self.source_lines, self.destination_lines, t)
        src_img = np.empty_like(self.source_origin)
        des_img = np.empty_like(self.destination_origin)
        source_wrap = self.warping(self.source_origin, self.source_lines, inter_lines)
        destination_wrap = self.warping(self.destination_origin, self.destination_lines, inter_lines)
        img = np.empty_like(self.source_origin)
        h,w,_ = self.source_origin.shape
        for j in range(w):
            for k in range(h):
                img[k, j] = (1 - t) * source_wrap[k, j] + t * destination_wrap[k, j]

        cv2.imwrite("source_wrap.jpg", source_wrap)
        cv2.imwrite("destination_wrap.jpg", destination_wrap)
        cv2.imwrite("result.jpg", img)

        src_wrap = QPixmap('source_wrap.jpg')
        des_wrap = QPixmap('destination_wrap.jpg')
        result  = QPixmap('result.jpg')
        self.sourceWarp.setPixmap(src_wrap)
        self.destinationWarp.setPixmap(des_wrap)
        self.resultimg.setPixmap(result)

        print("finish")

    def finDraw(self):
        cv2.imwrite('Source.jpg', self.source_img)
        cv2.imwrite('Destination.jpg',self.destination_img)
        cv2.destroyWindow("Source Image")
        cv2.destroyWindow("Destination Image")
        src = QPixmap('Source.jpg')
        des = QPixmap('Destination.jpg')
        self.sourceimg.setPixmap(src)
        self.destinationimg.setPixmap(des)

    def animate(self):
        animate = []
        for i in range(11):
            animate.append(cv2.imread('./result/frame__'+str(i+1)+'.jpg'))
        for img in animate:
            cv2.imshow('Animate', img)
            cv2.waitKey(300)



if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)
    window = mywindow()
    window.show()
    sys.exit(app.exec_())
