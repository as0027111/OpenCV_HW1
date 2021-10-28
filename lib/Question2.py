import cv2
import numpy as np
import glob

class Q2():
    def __init__(self) -> None:
        self.images = sorted(glob.glob('.\Dataset_CvDl_Hw1\Q2_Image\*.bmp'))
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        #axis =  np.float32([[2,3,0], [6,5,0], [6,1,0], [4,3,-3]]).reshape(-1,3) # 金字塔
        w, h = 8, 11        
        self.objp = np.zeros((w*h,3), np.float32)
        self.objp[:,:2] = np.mgrid[0:w,0:h].T.reshape(-1,2)
        self.objpoints = []
        self.imgpoints = []

        # do calibration for finding rvecs and tvecx matrix 
        for img_name in self.images:
            img = cv2.imread(img_name)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (w,h),None) #找棋盤方格
            if ret == True:
                self.objpoints.append(self.objp)
                self.imgpoints.append(corners)
        ret,self.mtx,self.dist,self.rvecs,self.tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, gray.shape[::-1], None, None)

    def Q2_1(self, word='OPENCV'):
        ch = word_str2ary(word)
        for index, fname in enumerate(self.images):
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (8,11),None)
            #print(rvecs[index],tvecs[index])
            if ret == True:
                corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),self.criteria)
                imgpts, jac = cv2.projectPoints(ch,self.rvecs[index],self.tvecs[index],self.mtx,self.dist)
                img = self.draw(img,corners2,imgpts)
                cv2.namedWindow(fname,cv2.WINDOW_NORMAL) 
                cv2.resizeWindow(fname,720,540)
                cv2.imshow(fname,img)
                cv2.waitKey(1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def Q2_2(self, word='OPENCV'): #Augmented Reality - Vertically
        ch = word_str2ary_vertical(word)
        # print(ch)
        for index, fname in enumerate(self.images):
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (8,11),None)
            #print(rvecs[index],tvecs[index])
            if ret == True:
                corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),self.criteria)
                imgpts, jac = cv2.projectPoints(ch,self.rvecs[index],self.tvecs[index],self.mtx,self.dist)
                img = self.draw(img,corners2,imgpts)
                cv2.namedWindow(fname,cv2.WINDOW_NORMAL) 
                cv2.resizeWindow(fname,720,540)
                cv2.imshow(fname,img)
                cv2.waitKey(1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def draw(self, img, corners, imgpts):
        imgpts = np.int32(imgpts).reshape(-1,2)
        #imgpts = np.array(list(combinations(imgpts,2)))
        for i in range(1, len(imgpts), 2): # 因為有多加一個 [0,0,0]
            img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[i+1]),(0,0,255),10)
        return img

def word_str2ary(words):
    matr_list = np.array([[2, 7, 0], [2, 4, 0], [2, 1, 0], [5, 7, 0], [5, 4, 0], [5, 1, 0], [8, 7, 0], [8, 4, 0], [8, 1, 0]]) # 字出現的位置
    word_list = list(words)
    font_dict = cv2.FileStorage('./Dataset_CvDl_Hw1/Q2_Image/Q2_Lib/alphabet_lib_onboard.txt', cv2.FILE_STORAGE_READ)
    return_list = np.array([0, 0, 0]) # 第一個記得要刪掉?
    for i, word in enumerate(word_list):
        # print(i, word)
        ch = font_dict.getNode(word).mat()
        test = np.float32(ch).reshape(-1, 3)
        # print(test)
        test = matr_list[i] + transform(test)
        return_list = np.row_stack((return_list, test))
    return return_list

def transform(array):
    for i in range(len(array)):
        array[i] = [-array[i,1], array[i,0], array[i,2]]
    return array

def word_str2ary_vertical(words):
    matr_list = np.array([[2, 7, 0], [2, 4, 0], [2, 1, 0], [5, 7, 0], [5, 4, 0], [5, 1, 0], [8, 7, 0], [8, 4, 0], [8, 1, 0]])
    word_list = list(words)
    font_dict = cv2.FileStorage('./Dataset_CvDl_Hw1/Q2_Image/Q2_Lib/alphabet_lib_onboard.txt', cv2.FILE_STORAGE_READ)
    return_list = np.array([0, 0, 0]) # 第一個記得要刪掉?
    for i, word in enumerate(word_list):
        # print(i, word)
        ch = font_dict.getNode(word).mat()
        test = np.float32(ch).reshape(-1, 3)
        test = matr_list[i] + trans2Vertical(test)
        return_list = np.row_stack((return_list, test))
    return return_list

def trans2Vertical(array):
    for i in range(len(array)):
        array[i] = [-array[i,2], array[i,0], -array[i,1]] # wanted version
        # array[i] = [-array[i,1], array[i,2], -array[i,0]] # more beautiful 
    return array

if __name__ == "__main__":  
    Q2 = Q2()
    Q2.Q2_1()
    Q2.Q2_2()