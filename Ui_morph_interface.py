# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'c:\Users\phant\Documents\NCKU\CG\HW2\HW2\morph_interface.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(781, 513)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(0, 40, 781, 191))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.sourceimg = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.sourceimg.setMinimumSize(QtCore.QSize(255, 189))
        self.sourceimg.setMaximumSize(QtCore.QSize(255, 189))
        self.sourceimg.setText("")
        self.sourceimg.setObjectName("sourceimg")
        self.horizontalLayout.addWidget(self.sourceimg)
        self.resultimg = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.resultimg.setMinimumSize(QtCore.QSize(255, 189))
        self.resultimg.setMaximumSize(QtCore.QSize(255, 189))
        self.resultimg.setText("")
        self.resultimg.setObjectName("resultimg")
        self.horizontalLayout.addWidget(self.resultimg)
        self.destinationimg = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.destinationimg.setMinimumSize(QtCore.QSize(255, 189))
        self.destinationimg.setMaximumSize(QtCore.QSize(255, 189))
        self.destinationimg.setText("")
        self.destinationimg.setObjectName("destinationimg")
        self.horizontalLayout.addWidget(self.destinationimg)
        self.horizontalLayoutWidget_2 = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget_2.setGeometry(QtCore.QRect(0, 0, 781, 41))
        self.horizontalLayoutWidget_2.setObjectName("horizontalLayoutWidget_2")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_2)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_3 = QtWidgets.QLabel(self.horizontalLayoutWidget_2)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_2.addWidget(self.label_3)
        self.label_2 = QtWidgets.QLabel(self.horizontalLayoutWidget_2)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_2.addWidget(self.label_2)
        self.label = QtWidgets.QLabel(self.horizontalLayoutWidget_2)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.horizontalLayout_2.addWidget(self.label)
        self.gridLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(530, 250, 116, 201))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.Getb = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.Getb.setObjectName("Getb")
        self.gridLayout.addWidget(self.Getb, 2, 1, 1, 1)
        self.GetT = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.GetT.setObjectName("GetT")
        self.gridLayout.addWidget(self.GetT, 0, 1, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.gridLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(9)
        self.label_4.setFont(font)
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 0, 0, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.gridLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(9)
        self.label_5.setFont(font)
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setObjectName("label_5")
        self.gridLayout.addWidget(self.label_5, 1, 0, 1, 1)
        self.Geta = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.Geta.setObjectName("Geta")
        self.gridLayout.addWidget(self.Geta, 1, 1, 1, 1)
        self.label_6 = QtWidgets.QLabel(self.gridLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(9)
        self.label_6.setFont(font)
        self.label_6.setAlignment(QtCore.Qt.AlignCenter)
        self.label_6.setObjectName("label_6")
        self.gridLayout.addWidget(self.label_6, 2, 0, 1, 1)
        self.label_7 = QtWidgets.QLabel(self.gridLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(9)
        self.label_7.setFont(font)
        self.label_7.setAlignment(QtCore.Qt.AlignCenter)
        self.label_7.setObjectName("label_7")
        self.gridLayout.addWidget(self.label_7, 3, 0, 1, 1)
        self.Getp = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.Getp.setObjectName("Getp")
        self.gridLayout.addWidget(self.Getp, 3, 1, 1, 1)
        self.Morphing = QtWidgets.QPushButton(self.centralwidget)
        self.Morphing.setGeometry(QtCore.QRect(540, 460, 93, 28))
        self.Morphing.setObjectName("Morphing")
        self.horizontalLayoutWidget_3 = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget_3.setGeometry(QtCore.QRect(0, 300, 519, 191))
        self.horizontalLayoutWidget_3.setObjectName("horizontalLayoutWidget_3")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_3)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.destinationWarp = QtWidgets.QLabel(self.horizontalLayoutWidget_3)
        self.destinationWarp.setMinimumSize(QtCore.QSize(255, 189))
        self.destinationWarp.setMaximumSize(QtCore.QSize(255, 189))
        self.destinationWarp.setText("")
        self.destinationWarp.setObjectName("destinationWarp")
        self.horizontalLayout_3.addWidget(self.destinationWarp)
        self.sourceWarp = QtWidgets.QLabel(self.horizontalLayoutWidget_3)
        self.sourceWarp.setMinimumSize(QtCore.QSize(255, 189))
        self.sourceWarp.setMaximumSize(QtCore.QSize(255, 189))
        self.sourceWarp.setText("")
        self.sourceWarp.setObjectName("sourceWarp")
        self.horizontalLayout_3.addWidget(self.sourceWarp)
        self.horizontalLayoutWidget_4 = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget_4.setGeometry(QtCore.QRect(0, 250, 521, 41))
        self.horizontalLayoutWidget_4.setObjectName("horizontalLayoutWidget_4")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_4)
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.label_9 = QtWidgets.QLabel(self.horizontalLayoutWidget_4)
        self.label_9.setAlignment(QtCore.Qt.AlignCenter)
        self.label_9.setObjectName("label_9")
        self.horizontalLayout_4.addWidget(self.label_9)
        self.label_8 = QtWidgets.QLabel(self.horizontalLayoutWidget_4)
        self.label_8.setAlignment(QtCore.Qt.AlignCenter)
        self.label_8.setObjectName("label_8")
        self.horizontalLayout_4.addWidget(self.label_8)
        self.Draw = QtWidgets.QPushButton(self.centralwidget)
        self.Draw.setGeometry(QtCore.QRect(660, 300, 101, 28))
        self.Draw.setObjectName("Draw")
        self.Show = QtWidgets.QPushButton(self.centralwidget)
        self.Show.setGeometry(QtCore.QRect(660, 400, 101, 31))
        self.Show.setObjectName("Show")
        self.Finish = QtWidgets.QPushButton(self.centralwidget)
        self.Finish.setGeometry(QtCore.QRect(660, 350, 101, 28))
        self.Finish.setObjectName("Finish")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_3.setText(_translate("MainWindow", "Source Image"))
        self.label_2.setText(_translate("MainWindow", "Result Image"))
        self.label.setText(_translate("MainWindow", "Destination Image"))
        self.label_4.setText(_translate("MainWindow", "   T:  "))
        self.label_5.setText(_translate("MainWindow", " a:"))
        self.label_6.setText(_translate("MainWindow", " b:"))
        self.label_7.setText(_translate("MainWindow", " p:"))
        self.Morphing.setText(_translate("MainWindow", "Morphing"))
        self.label_9.setText(_translate("MainWindow", "Source Warped"))
        self.label_8.setText(_translate("MainWindow", "Destination Warped"))
        self.Draw.setText(_translate("MainWindow", "DrawLine"))
        self.Show.setText(_translate("MainWindow", "ShowAnimate"))
        self.Finish.setText(_translate("MainWindow", "FinishDraw"))
