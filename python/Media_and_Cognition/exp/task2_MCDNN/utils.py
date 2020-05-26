'''
@Description:
@Author: Zhouyx
该文件用于存放一些标准化的函数
'''
import numpy as np
import cv2
import os
def Imadjust(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    img_shape = np.shape(img)
    num_of_pixels = img_shape[0]*img_shape[1]
    new_img = np.zeros(shape=img_shape, dtype=np.uint8)

    histogram = np.zeros((256,1))
    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            histogram[img[i,j,0]] += 1
    cnt = 0
    for i in range(256):
        if cnt < round(0.01*num_of_pixels):
            pass
        elif cnt >round(0.99*num_of_pixels):
            new_img[:,:,0] += ((img[:,:,0]==i).astype(np.uint8)*255).astype(np.uint8)
        else:
            new_img[:,:,0] += ((img[:,:,0] == i).astype(np.uint8) * i).astype(np.uint8)
        cnt += histogram[i]
    new_img[:,:,1] = img[:,:,1]
    new_img[:, :, 2] = img[:, :, 2]
    new_img = cv2.cvtColor(new_img, cv2.COLOR_Lab2BGR)
    return new_img


def Histeq(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    img[:,:,0] = cv2.equalizeHist(img[:,:,0])
    img = cv2.cvtColor(img, cv2.COLOR_Lab2BGR)
    return img

def Adaphisteq(img):
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(6, 6))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    img[:,:,0] = clahe.apply(img[:,:,0])
    img = cv2.cvtColor(img, cv2.COLOR_Lab2BGR)
    return img


def Conorm(img):
    img1 = cv2.GaussianBlur(img,(5,5),1.0)
    img2 = cv2.GaussianBlur(img, (5, 5), 2.0)
    return img1-img2


def load_images(path,labels,is_train):
    data_name = []
    if is_train:
        data = {}
        for label in labels:
            tmp = []
            for filename in os.listdir(path+label+'/'):
                img = cv2.imread(path+label+'/'+filename)
                tmp.append(img)
                data_name.append(filename)
            data[label] = tmp
    else:
        data = []
        for filename in os.listdir(path):
            img = cv2.imread(path+ filename)
            data.append(img)
            data_name.append(filename)
    return data, data_name
