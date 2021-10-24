import cv2
import numpy as np
def Show_image(Window_name, width, height, img):
    cv2.namedWindow(Window_name, cv2.WINDOW_NORMAL) 
    cv2.resizeWindow(Window_name, width,height)
    cv2.imshow(Window_name, img)


