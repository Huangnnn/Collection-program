# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'face_multicamera_mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1920, 1080)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label_show = QtWidgets.QLabel(self.centralwidget)
        self.label_show.setGeometry(QtCore.QRect(0, 0, 1920, 720))
        self.label_show.setObjectName("label_show")
        self.pushButton_record_start = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_record_start.setGeometry(QtCore.QRect(330, 750, 101, 41))
        self.pushButton_record_start.setObjectName("pushButton_record_start")
        self.lineEdit_id = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_id.setGeometry(QtCore.QRect(140, 750, 151, 41))
        self.lineEdit_id.setObjectName("lineEdit_id")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(50, 760, 61, 21))
        self.label.setObjectName("label")
        self.pushButton_record_stop = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_record_stop.setGeometry(QtCore.QRect(470, 750, 101, 41))
        self.pushButton_record_stop.setObjectName("pushButton_record_stop")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(70, 971, 54, 21))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(210, 971, 54, 21))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(340, 971, 81, 21))
        self.label_4.setObjectName("label_4")
        self.label_frame = QtWidgets.QLabel(self.centralwidget)
        self.label_frame.setGeometry(QtCore.QRect(120, 970, 61, 21))
        self.label_frame.setObjectName("label_frame")
        self.label_time = QtWidgets.QLabel(self.centralwidget)
        self.label_time.setGeometry(QtCore.QRect(260, 970, 61, 21))
        self.label_time.setObjectName("label_time")
        self.label_frame_rate_avg = QtWidgets.QLabel(self.centralwidget)
        self.label_frame_rate_avg.setGeometry(QtCore.QRect(430, 970, 61, 21))
        self.label_frame_rate_avg.setObjectName("label_frame_rate_avg")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(490, 970, 81, 21))
        self.label_5.setObjectName("label_5")
        self.label_frame_rate_rt = QtWidgets.QLabel(self.centralwidget)
        self.label_frame_rate_rt.setGeometry(QtCore.QRect(580, 970, 91, 21))
        self.label_frame_rate_rt.setObjectName("label_frame_rate_rt")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1920, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "3D Face Acquisition"))
        self.label_show.setText(_translate("MainWindow", "Depth"))
        self.pushButton_record_start.setText(_translate("MainWindow", "Start"))
        self.label.setText(_translate("MainWindow", "ID:"))
        self.pushButton_record_stop.setText(_translate("MainWindow", "Stop"))
        self.label_2.setText(_translate("MainWindow", "FRAME："))
        self.label_3.setText(_translate("MainWindow", "TIME："))
        self.label_4.setText(_translate("MainWindow", "FPS_AVG："))
        self.label_frame.setText(_translate("MainWindow", "frame"))
        self.label_time.setText(_translate("MainWindow", "time"))
        self.label_frame_rate_avg.setText(_translate("MainWindow", "avg"))
        self.label_5.setText(_translate("MainWindow", "FPS_RT:"))
        self.label_frame_rate_rt.setText(_translate("MainWindow", "real_time"))