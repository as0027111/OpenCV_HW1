import cv2
import numpy as np
from lib.Question5 import Q5
from lib.utilize import Show_image

class Q4:
    def __init__(self, img1, img2):
        self.img1 = cv2.imread(img1,0) 
        self.img2 = cv2.imread(img2,0) 
        # Initiate SIFT detector
        self.sift = cv2.SIFT_create()
        print("init")
    # Find keypoint, sorting and plot  
    def Q4_1(self): 
        keypoint_1 = self.sift.detect(self.img1)
        srt_1 = sorted(keypoint_1, key = lambda x:x.size, reverse=True)
        self.srt200_1 = srt_1[:200]
        # img_1 = cv2.drawKeypoints(self.img1,keypoint_1,np.array([]),color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        img_1 = cv2.drawKeypoints(self.img1,self.srt200_1,np.array([]),color=(0, 0, 255))

        keypoint_2 = self.sift.detect(self.img2)
        srt_2 = sorted(keypoint_2, key = lambda x:x.size, reverse=True)
        self.srt200_2 = srt_2[:200]
        img_2 = cv2.drawKeypoints(self.img2,self.srt200_2,np.array([]),color=(0, 255, 0))
        Show_image("img1", 720, 540, img_1)
        Show_image("img2", 720, 540, img_2)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return 

    def Q4_2(self):
        self.kpts_1, des1 = self.sift.compute(self.img1, self.srt200_1)
        self.kpts_2, des2 = self.sift.compute(self.img2, self.srt200_2)
        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary
        self.flann = cv2.FlannBasedMatcher(index_params,search_params)
        matches = self.flann.knnMatch(des2, des1, k=2)
    
        # ratio test as per Lowe's paper
        # https://stackoverflow.com/questions/46607647/sift-feature-matching-point-coordinates
        matchesMask = [[0,0] for i in range(len(matches))]
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.75*n.distance:
                matchesMask[i]=[1,0]
        draw_params = dict(matchColor = (0,255,0),
                        singlePointColor = (255,0,0),
                        matchesMask = matchesMask,
                        flags = 0)

        img3 = cv2.drawMatchesKnn(self.img2, self.kpts_2,self.img1, self.kpts_1,  matches, None, **draw_params)
        Show_image("Matching", 720*2, 540, img3)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return 

    def Q4_3(self):
        sift=cv2.SIFT_create()
        kp1,descrip1=sift.detectAndCompute(self.img1,None)
        kp2,descrip2=sift.detectAndCompute(self.img2,None)
        match=self.flann.knnMatch(descrip1,descrip2,k=2)
        good=[]
        for i,(m,n) in enumerate(match):
                if(m.distance<0.75*n.distance):
                        good.append(m)
        MIN = 10
        if len(good)>MIN:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
                ano_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
                M,mask=cv2.findHomography(src_pts,ano_pts,cv2.RANSAC,5.0)
                warpImg = cv2.warpPerspective(self.img2, np.linalg.inv(M), (self.img1.shape[1]+self.img2.shape[1], self.img2.shape[0]))
                direct=warpImg.copy()
                direct[0:self.img1.shape[0], 0:self.img1.shape[1]] = self.img1
        Show_image("Result", 720*2, 540, direct)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    

if __name__ == "__main__":  
    img_path_1 = './Dataset_CvDl_Hw1./Q4_Image./Shark1.jpg'
    img_path_2 = './Dataset_CvDl_Hw1./Q4_Image./Shark2.jpg'

    Q4 = Q4(img_path_1, img_path_2)
    Q4.Q4_1()
    Q4.Q4_2()
    Q4.Q4_3()




