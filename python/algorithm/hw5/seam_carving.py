'''
@Description: 
@Author: HuYi
@Date: 2020-04-06 13:46:04
@LastEditors: HuYi
@LastEditTime: 2020-04-06 13:47:38
'''
import cv2 as cv


# Sobel算子


def sobel_demo(img):
    grad_x = cv.Sobel(img, cv.CV_32F, 1, 0)  # 对x求一阶导
    grad_y = cv.Sobel(img, cv.CV_32F, 0, 1)  # 对y求一阶导
    gradx = cv.convertScaleAbs(grad_x)  # 用convertScaleAbs()函数将其转回原来的uint8形式
    grady = cv.convertScaleAbs(grad_y)
    cv.imshow("gradient_x", gradx)  # x方向上的梯度
    cv.imshow("gradient_y", grady)  # y方向上的梯度
    gradxy = cv.addWeighted(gradx, 0.5, grady, 0.5, 0)  # 图片融合
    cv.imshow("gradient", gradxy)


src = cv.imread('erzi.jpg')
cv.namedWindow('input_image', cv.WINDOW_NORMAL)  # 设置为WINDOW_NORMAL可以任意缩放
cv.imshow('input_image', src)
sobel_demo(src)
cv.waitKey(0)
cv.destroyAllWindows()
