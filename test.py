import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
MIN = 10
starttime=time.time()
img1 = cv2.imread('./Dataset_CvDl_Hw1./Q4_Image./Shark1.jpg') #query
img2 = cv2.imread('./Dataset_CvDl_Hw1./Q4_Image./Shark2.jpg') #train

#img1gray=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
#img2gray=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
sift=cv2.SIFT_create()
#surf=cv2.xfeatures2d.SIFT_create()#可以改為SIFT
kp1,descrip1=sift.detectAndCompute(img1,None)
kp2,descrip2=sift.detectAndCompute(img2,None)

FLANN_INDEX_KDTREE = 0
indexParams = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
searchParams = dict(checks=50)

flann=cv2.FlannBasedMatcher(indexParams,searchParams)
match=flann.knnMatch(descrip1,descrip2,k=2)


good=[]
for i,(m,n) in enumerate(match):
        if(m.distance<0.75*n.distance):
                good.append(m)

if len(good)>MIN:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        ano_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
        M,mask=cv2.findHomography(src_pts,ano_pts,cv2.RANSAC,5.0)
        warpImg = cv2.warpPerspective(img2, np.linalg.inv(M), (img1.shape[1]+img2.shape[1], img2.shape[0]))
        direct=warpImg.copy()
        direct[0:img1.shape[0], 0:img1.shape[1]] = img1

cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
cv2.imshow("Result",direct)
cv2.waitKey(0)
cv2.destroyAllWindows()