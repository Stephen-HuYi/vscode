'''
@Description: 
@Author: HuYi
@Date: 2020-06-16 22:13:00
@LastEditors: HuYi
@LastEditTime: 2020-06-16 22:35:29
'''


import cv2
import numpy as np


def wiener(input, PSF, K=0.01):  # 维纳滤波，K=0.01
    input_fft = np.fft.fft2(input)
    PSF_fft = np.fft.fft2(PSF)
    PSF_fft_1 = np.conj(PSF_fft) / (np.abs(PSF_fft)**2 + K)
    result = np.fft.ifft2(input_fft * PSF_fft_1)
    result = np.real(fft.fftshift(result))
    result[result > 255.0] = 255
    result[result < 0] = 0
    return result


root = 'D:/vscode/python/work/'
img = cv2.imread(root+'test.jpg', flags=2)
cv2.imshow('test', img)
cv2.waitKey(0)
