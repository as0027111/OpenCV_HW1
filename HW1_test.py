# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'HW1_main.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
import cv2
import glob
import numpy as np
import time 
from lib import Question3, Question4, Question2

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(768, 478)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(60, 40, 141, 381))
        self.groupBox.setObjectName("groupBox")
        self.groupBox_5 = QtWidgets.QGroupBox(self.groupBox)
        self.groupBox_5.setGeometry(QtCore.QRect(10, 150, 121, 80))
        self.groupBox_5.setObjectName("groupBox_5")
        self.pushButton_3 = QtWidgets.QPushButton(self.groupBox_5)
        self.pushButton_3.setGeometry(QtCore.QRect(10, 52, 101, 21))
        self.pushButton_3.setObjectName("pushButton_3")
        self.label = QtWidgets.QLabel(self.groupBox_5)
        self.label.setGeometry(QtCore.QRect(10, 20, 59, 12))
        self.label.setObjectName("label")
        self.comboBox = QtWidgets.QComboBox(self.groupBox_5)
        self.comboBox.setGeometry(QtCore.QRect(70, 20, 41, 22))
        self.comboBox.setObjectName("comboBox")

        self.pushButton = QtWidgets.QPushButton(self.groupBox)
        self.pushButton.setGeometry(QtCore.QRect(11, 49, 121, 23))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_2.setGeometry(QtCore.QRect(11, 110, 121, 23))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_4 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_4.setGeometry(QtCore.QRect(11, 250, 121, 23))
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_5 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_5.setGeometry(QtCore.QRect(11, 317, 121, 23))
        self.pushButton_5.setObjectName("pushButton_5")
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(220, 40, 151, 381))
        self.groupBox_2.setObjectName("groupBox_2")
        self.layoutWidget = QtWidgets.QWidget(self.groupBox_2)
        self.layoutWidget.setGeometry(QtCore.QRect(9, 10, 136, 371))
        self.layoutWidget.setObjectName("layoutWidget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.layoutWidget)
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.lineEdit = QtWidgets.QLineEdit(self.layoutWidget)
        self.lineEdit.setObjectName("lineEdit")
        self.lineEdit.setText("OPENCV")
        self.gridLayout_2.addWidget(self.lineEdit, 0, 0, 1, 1)
        self.pushButton_6 = QtWidgets.QPushButton(self.layoutWidget)
        self.pushButton_6.setObjectName("pushButton_6")
        self.gridLayout_2.addWidget(self.pushButton_6, 1, 0, 1, 1)
        self.pushButton_7 = QtWidgets.QPushButton(self.layoutWidget)
        self.pushButton_7.setObjectName("pushButton_7")
        self.gridLayout_2.addWidget(self.pushButton_7, 2, 0, 1, 1)
        self.groupBox_3 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_3.setGeometry(QtCore.QRect(390, 40, 151, 381))
        self.groupBox_3.setObjectName("groupBox_3")
        self.pushButton_8 = QtWidgets.QPushButton(self.groupBox_3)
        self.pushButton_8.setGeometry(QtCore.QRect(10, 180, 131, 23))
        self.pushButton_8.setObjectName("pushButton_8")
        self.groupBox_4 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_4.setGeometry(QtCore.QRect(560, 40, 151, 381))
        self.groupBox_4.setObjectName("groupBox_4")
        self.layoutWidget1 = QtWidgets.QWidget(self.groupBox_4)
        self.layoutWidget1.setGeometry(QtCore.QRect(10, 10, 131, 371))
        self.layoutWidget1.setObjectName("layoutWidget1")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.layoutWidget1)
        self.gridLayout_3.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.pushButton_9 = QtWidgets.QPushButton(self.layoutWidget1)
        self.pushButton_9.setObjectName("pushButton_9")
        self.gridLayout_3.addWidget(self.pushButton_9, 0, 0, 1, 1)
        self.pushButton_11 = QtWidgets.QPushButton(self.layoutWidget1)
        self.pushButton_11.setObjectName("pushButton_11")
        self.gridLayout_3.addWidget(self.pushButton_11, 2, 0, 1, 1)
        self.pushButton_10 = QtWidgets.QPushButton(self.layoutWidget1)
        self.pushButton_10.setObjectName("pushButton_10")
        self.gridLayout_3.addWidget(self.pushButton_10, 1, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 768, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        information = [str(i) for i in range(1,16)]
        self.comboBox.addItems(information)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.Q2 = Question2.Q2()

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBox.setTitle(_translate("MainWindow", "1. Calibration"))
        self.groupBox_5.setTitle(_translate("MainWindow", "1.3 Find Extrinsic"))
        self.pushButton_3.setText(_translate("MainWindow", "1.3 Find Extrinsic"))
        self.pushButton_3.clicked.connect(self.B1_3)
        self.label.setText(_translate("MainWindow", "Select Image"))
        self.pushButton.setText(_translate("MainWindow", "1.1 Find Corners"))
        self.pushButton.clicked.connect(self.B1_1)
        self.pushButton_2.setText(_translate("MainWindow", "1.2 Find Intrinsic"))
        self.pushButton_2.clicked.connect(self.B1_2)
        self.pushButton_4.setText(_translate("MainWindow", "1.4 Find Distortion"))
        self.pushButton_4.clicked.connect(self.B1_4)
        self.pushButton_5.setText(_translate("MainWindow", "1.5 Show Result"))
        self.pushButton_5.clicked.connect(self.B1_5)
        self.groupBox_2.setTitle(_translate("MainWindow", "2. Augmented Realty"))
        self.pushButton_6.setText(_translate("MainWindow", "2.1 Show Words on Board"))
        self.pushButton_6.clicked.connect(self.B2_1)
        self.pushButton_7.setText(_translate("MainWindow", "2.2 Show Words Vertically"))
        self.pushButton_7.clicked.connect(self.B2_2)
        self.groupBox_3.setTitle(_translate("MainWindow", "3. Stereo Disparity Map"))
        self.pushButton_8.setText(_translate("MainWindow", "3.1 Stereo Disparity Map"))
        self.pushButton_8.clicked.connect(self.B3_1)

        self.groupBox_4.setTitle(_translate("MainWindow", "4. SIFT"))
        self.pushButton_9.setText(_translate("MainWindow", "4.1 Keypoints"))
        self.pushButton_9.clicked.connect(self.B4_1)
        self.pushButton_10.setText(_translate("MainWindow", "4.2 Matched Keypoints"))
        self.pushButton_10.clicked.connect(self.B4_2)        
        self.pushButton_11.setText(_translate("MainWindow", "4.3 Wrap Image"))
        self.pushButton_11.clicked.connect(self.B4_3)


    def B1_1(self, MainWindow): #Find Corners
        w, h = 8, 11
        self.objp = np.zeros((w*h,3), np.float32)
        self.objp[:,:2] = np.mgrid[0:w,0:h].T.reshape(-1,2)

        self.objpoints = [] # 在世界座標系中的三維座標
        self.imgpoints = [] # 在圖像平面的二維座標    
        
        self.images = sorted(glob.glob('.\Dataset_CvDl_Hw1\Q1_Image\*.bmp'))
        print(self.images)
        w, h = 8, 11 
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)#閾值

        for img_name in self.images:
            img = cv2.imread(img_name)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (w,h),None) #找棋盤方格
            if ret == True:
                self.objpoints.append(self.objp)
                self.imgpoints.append(corners)
                cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
                cv2.drawChessboardCorners(img, (w,h), corners, ret) #畫方格
                cv2.namedWindow(img_name,cv2.WINDOW_NORMAL) 
                cv2.resizeWindow(img_name,720,540)
                cv2.imshow(img_name,img)
                cv2.waitKey(1)
                # time.sleep(0.5)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def B1_2(self, MainWindow):  #Find Intrinsic
        gray = cv2.imread('./Dataset_CvDl_Hw1/Q1_Image/1.bmp', 0)
        ret,self.mtx,self.dist,self.rvecs,self.tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, gray.shape[::-1], None, None)
        print(self.mtx)

    def B1_3(self, MainWindow):  #Find Extrinsic
        order_list = [0,7,8,9,10,11,12,13,14,1,2,3,4,5,6]
        num = int(self.comboBox.currentText())-1
        num = order_list[num]
        #print(num)
        rotational_matrix, _ = cv2.Rodrigues(self.rvecs[num]) 
        Extrinsic_matrix = np.concatenate((rotational_matrix,self.tvecs[num]),axis=1)
        print(Extrinsic_matrix)

    def B1_4(self, MainWindow):  #Find Distortion
        print(self.dist)

    def B1_5(self, MainWindow):  #Fix Distortion    
        for img_name in self.images:
            img1 = cv2.imread(img_name)
            h,  w = img1.shape[:2]
            newcameramtx, roi=cv2.getOptimalNewCameraMatrix(self.mtx,self.dist,(w,h),0,(w,h)) 
            dst = cv2.undistort(img1, self.mtx, self.dist, None, newcameramtx)
            # 根據前面ROI區域裁剪圖片
            #x,y,w,h = roi
            #dst = dst[y:y+h, x:x+w]
            show_2pic_stack = np.concatenate((img1, dst), axis=1)
            cv2.namedWindow("distortion",cv2.WINDOW_NORMAL) 
            cv2.resizeWindow("distortion",720*2,540)
            cv2.imshow("distortion",show_2pic_stack)
            cv2.waitKey(1)
            time.sleep(0.5)
            cv2.destroyAllWindows()

    def B2_1(self, MainWindow): #Augmented Reality - On Board
        word = self.lineEdit.text() 
        
        self.Q2.Q2_1(word)

    def B2_2(self, MainWindow): #Augmented Reality - Vertically
        word = self.lineEdit.text() 
        self.Q2.Q2_2(word)

    def B3_1(self, MainWindow):
        Q3 = Question3.Q3()
        Q3.Q3_1()

    def B4_1(self, MainWindow):
        img_path_1 = './Dataset_CvDl_Hw1./Q4_Image./Shark1.jpg'
        img_path_2 = './Dataset_CvDl_Hw1./Q4_Image./Shark2.jpg'
        self.Q4 = Question4.Q4(img_path_1, img_path_2)
        self.Q4.Q4_1()

    def B4_2(self, MainWindow):
        self.Q4.Q4_2()

    def B4_3(self, MainWindow):
        self.Q4.Q4_3()

        


if __name__ == "__main__":  
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
