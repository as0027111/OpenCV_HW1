import cv2
import numpy as np
class Q3:
    def Q3_1(self): 
        self.L = cv2.imread('./Dataset_CvDL_Hw1/Q3_Image/imL.png') # 用來顯示圖片
        self.R = cv2.imread('./Dataset_CvDL_Hw1/Q3_Image/imR.png')
        imgL = cv2.imread('./Dataset_CvDL_Hw1/Q3_Image/imL.png', 0) # 用來做 disparity
        imgR = cv2.imread('./Dataset_CvDL_Hw1/Q3_Image/imR.png', 0)

        #Block Size: Must be odd and within the range [5, 255]
        #Disparity range: Must be positive and divisible by 16
        stereo = cv2.StereoBM_create(numDisparities=256, blockSize=25)
        
        self.disparity = stereo.compute(imgL,imgR)
        min = self.disparity.min()
        max = self.disparity.max()
        # print(min, max)
        disparity_SGBM = cv2.normalize(self.disparity, self.disparity, alpha=255,
                              beta=0, norm_type=cv2.NORM_MINMAX)
        self.disparity_SGBM = np.uint8(disparity_SGBM)

        cv2.namedWindow("L",cv2.WINDOW_NORMAL) 
        cv2.resizeWindow("L", 720,540)
        cv2.imshow('L', self.L)
        cv2.setMouseCallback('L', self.draw_circle) # 綁定點擊的事件

        cv2.namedWindow("R",cv2.WINDOW_NORMAL) 
        cv2.resizeWindow("R", 720,540)
        cv2.imshow('R', self.R)

        cv2.namedWindow("Disparity",cv2.WINDOW_NORMAL) 
        cv2.resizeWindow("Disparity", 720,540)
        cv2.imshow("Disparity", self.disparity_SGBM)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def draw_circle(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            print("Dis", self.disparity[y,x])
            dis = self.disparity[y,x]
            
            img = self.L.copy()
            cv2.circle(img, (x, y), 10, (0, 0, 255), -1)
            cv2.imshow('L', img)       
            img =self.R.copy()     
            cv2.circle(img, (x - dis, y), 10, (0, 0, 255), -1)
            cv2.imshow('R', img)
            #print(x,y)
            
if __name__ == "__main__":  
    Q3 = Q3()
    Q3.Q3_1()
    