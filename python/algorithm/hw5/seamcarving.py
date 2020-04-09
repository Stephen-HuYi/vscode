'''
@Description: 
@Author: HuYi
@Date: 2020-04-04 22:22:46
@LastEditors: HuYi
@LastEditTime: 2020-04-06 22:28:41
'''
import sys
import numpy as np
from imageio import imread, imwrite
from scipy.ndimage.filters import convolve
from numba import jit


def get_d(img):
    # Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  #soble算子
    Gx = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]])  # scharr算子
    Gy = np.transpose(Gx)
    # 得到3D滤波器
    Gx = np.stack([Gx] * 3, axis=2)
    Gy = np.stack([Gy] * 3, axis=2)
    img = img.astype('float32')
    convolved = np.absolute(convolve(img, Gx)) + np.absolute(convolve(img, Gy))
    d = convolved.sum(axis=2)
    return d


@jit
def del_seam_col(img):
    r, c, _ = img.shape
    d = get_d(img)
    M = d.copy()
    seam = np.zeros((r, c), dtype=np.int)
    for i in range(1, r):
        for j in range(0, c):
            if j == 0:
                seam[i, j] = np.argmin(M[i - 1, j:j + 2]) + j
                d_min = M[i - 1, seam[i, j]]
            else:
                seam[i, j] = np.argmin(M[i - 1, j - 1:j + 2]) + j - 1
                d_min = M[i - 1, seam[i, j]]
            M[i, j] += d_min
    mask = np.ones((r, c), dtype=np.bool)
    j = np.argmin(M[-1])
    for i in range(r-1, -1, -1):
        mask[i, j] = False
        j = seam[i, j]
    mask = np.stack([mask] * 3, axis=2)
    img = img[mask].reshape((r, c - 1, 3))
    return img


print('Please enter the file name in the directory at the same level')
img = imread(input())
r, c, _ = img.shape
for i in range(int(0.5 * c)):
    img = del_seam_col(img)
img = np.rot90(img, 1, (0, 1))  # 转置后进行每列删除一个的操作
for i in range(int(0.5 * r)):
    img = del_seam_col(img)
img = np.rot90(img, 3, (0, 1))
print('Please enter the file name of the result')
imwrite(input(), img)
print('Compression succeeded! ! !')
os.system("pause")
