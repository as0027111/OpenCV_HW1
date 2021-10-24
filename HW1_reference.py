import cv2
import numpy as np

def word_str2ary(words):
    matr_list = np.array([[2, 7, 0], [2, 4, 0], [2, 1, 0], [5, 7, 0], [5, 4, 0], [5, 1, 0], [8, 7, 0], [8, 4, 0], [8, 1, 0]]) # 字出現的位置
    word_list = list(words)
    font_dict = cv2.FileStorage('./Dataset_CvDl_Hw1/Q2_Image/Q2_Lib/alphabet_lib_onboard.txt', cv2.FILE_STORAGE_READ)
    return_list = np.array([0, 0, 0]) # 第一個記得要刪掉?
    for i, word in enumerate(word_list):
        # print(i, word)
        ch = font_dict.getNode(word).mat()
        test = np.float32(ch).reshape(-1, 3)
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

def Show_image(Window_name, width, height, img):
    cv2.namedWindow(Window_name, cv2.WINDOW_NORMAL) 
    cv2.resizeWindow(Window_name, width,height)
    cv2.imshow(Window_name, img)
