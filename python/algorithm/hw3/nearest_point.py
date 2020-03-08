'''
@Description: 
@Author: HuYi
@Date: 2020-03-08 11:24:32
@LastEditors: HuYi
@LastEditTime: 2020-03-08 15:33:32
'''
import random
import math
from tkinter import *

# 计算两点间距离


def dis(array):
    dis = math.sqrt((array[0][0]-array[1][0])**2+(array[0][1]-array[1][1])**2)
    return dis

# 生成器：生成横跨跨两个点集的候选点


def candidateDot(u, right, dis, med_x):
    # 遍历right（已按横坐标升序排序）。若横坐标小于med_x-dis则进入下一次循环；若横坐标大于med_x+dis则跳出循环；若点的纵坐标好是否落在在[u[1]-dis,u[1]+dis]，则返回这个点
    for v in right:
        if v[0] < med_x-dis:
            continue
        if v[0] > med_x+dis:
            break
        if v[1] >= u[1]-dis and v[1] <= u[1]+dis:
            yield v

# 求出横跨两个部分的点的最小距离


def combine(left, right, res_min, med_x):
    dis = res_min[1]
    dis_min = res_min[1]
    pair = res_min[0]
    for u in left:
        if u[0] < med_x - dis:
            continue
        for v in candidateDot(u, right, dis, med_x):
            dis = dis([u, v])
            if dis < dis_min:
                dis_min = dis
                pair = [u, v]
    return [pair, dis_min]


# 分治求解
def divide(array):
    # 求序列元素数量
    n = len(array)
    # 按点的纵坐标升序排序
    array = sorted(array)
    # 递归开始进行
    if n <= 1:
        return None, float('inf')
    elif n == 2:
        return [array, dis(array)]
    else:
        half = int(len(array)/2)
        med_x = (array[half][0]+array[-half-1][0])/2
        left = array[:half]
        resLeft = divide(left)
        right = array[half:]
        resRight = divide(right)
        # 获取两集合中距离最短的点对
        if resLeft[1] < resRight[1]:
            resMin = combine(left, right, resLeft, med_x)
        else:
            resMin = combine(left, right, resRight, med_x)
        pair = resMin[0]
        minDis = resMin[1]
    return [pair, minDis]


def callback(event):
    print("当前位置：", event.x, event.y)


app = Tk()
# 创建框架，窗口尺寸
frame = Frame(app, width=400, height=400)
# frame.bind("<Motion>",callback)
frame.bind("<Button-1>", callback)
frame.bind("<Button-2>", callback)
frame.bind("<Button-3>", callback)
frame.pack()
# <Button-1>Button：表示鼠标的点击事件 “—”左边是事件本身，右边是事件描述
# 1：表示左键 2：中间键的滚轮点击 3：右键

mainloop()
array = [(random.randint(0, 100), random.randint(0, 100))
         for x in range(500000)]
print("优化算法", divide(array))
