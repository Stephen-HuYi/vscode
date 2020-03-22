'''
@Description:
@Author: HuYi
@Date: 2020-03-08 11:24:32
@LastEditors: HuYi
@LastEditTime: 2020-03-12 11:36:07
'''
import random
import math
import time
import tkinter as tk


def get_dis(array):
    return math.sqrt((array[0][0]-array[1][0])**2+(array[0][1]-array[1][1])**2)


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
        if u[0] < med_x-dis:
            continue
        for v in candidateDot(u, right, dis, med_x):
            dis = get_dis([u, v])
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
        return [array, get_dis(array)]
    else:
        half = int(len(array)/2)
        med_x = (array[half][0]+array[-half-1][0])/2
        left = array[:half]
        res_left = divide(left)
        right = array[half:]
        res_right = divide(right)
        # 获取两集合中距离最短的点对
        if res_left[1] < res_right[1]:
            res_min = combine(left, right, res_left, med_x)
        else:
            res_min = combine(left, right, res_right, med_x)
        pair = res_min[0]
        dis_min = res_min[1]
    return [pair, dis_min]


# 图形界面实现
window = tk.Tk()
window.title('window')
window.geometry('800x600')
show = tk.Message(window, text="计算结果(Θ(nlgn)算法)",
                  font=('Arial', 12), width=200)
show.pack()
text = tk.Text(window, width=100, height=6)
text.pack()
e = tk.Entry(window, show=None)
e.place(x=500, y=115, anchor='nw')
w = tk.Canvas(window, width=800, height=400, background='white')
w.place(x=0, y=185, anchor='nw')
array = []


def handle(event, array):
    array.append((event.x, event.y))
    x1, y1 = (event.x - 1), (event.y - 1)
    x2, y2 = (event.x + 1), (event.y + 1)
    w.create_oval(x1, y1, x2, y2, fill='red')


def moderandom():
    num = int(e.get())
    array = [(random.uniform(-10000, 10000), random.uniform(-10000, 10000))
             for x in range(num)]
    start = time.time()
    temp = divide(array)
    end = time.time()
    time_cost = end-start
    text.delete(1.0, tk.END)
    text.insert(1.0, "The result of moderandom:\n")
    text.insert(2.0, "nearest points:\n")
    text.insert(3.0, temp[0])
    text.insert(4.0, "\n")
    text.insert(4.0, "their distance:\n")
    text.insert(5.0, temp[1])
    text.insert(6.0, "\n")
    text.insert(6.0, "totally time cost = ")
    text.insert("6.end", time_cost)
    text.insert("6.end", " seconds")


def calc(array):
    text.delete(1.0, tk.END)
    start = time.time()
    temp = divide(array)
    end = time.time()
    time_cost = end-start
    text.delete(1.0, tk.END)
    text.insert(1.0, "The result of modemouse:\n")
    text.insert(2.0, "nearest points:\n")
    text.insert(3.0, temp[0])
    text.insert(4.0, "\n")
    text.insert(4.0, "their distance:\n")
    text.insert(5.0, temp[1])
    text.insert(6.0, "\n")
    text.insert(6.0, "totally time cost = ")
    text.insert("6.end", time_cost)
    text.insert("6.end", " seconds")


def modemouse(array):
    w.delete("all")
    array[:] = []
    w.bind("<Button-1>", lambda event: handle(event, array))
    w.bind("<Button-2>", lambda event: handle(event, array))
    w.bind("<Button-3>", lambda event: handle(event, array))


def deal(array):
    w.delete("all")
    array[:] = []


a = tk.Button(window, text='随机生成模式(在右侧方框内输入点的个数后点此按钮计算)', font=('Arial', 12),
              width=50, height=1, command=moderandom)
a.place(x=0, y=110, anchor='nw')
b = tk.Button(window, text='鼠标交互模式(点击此按钮后在下面方框内选点)', font=('Arial', 12),
              width=50, height=1, command=lambda: modemouse(array))
b.place(x=0, y=150, anchor='nw')
c = tk.Button(window, text='点击此按钮计算鼠标交互模式结果', font=('Arial', 12),
              width=30, height=1, command=lambda: calc(array))
c.place(x=500, y=150, anchor='nw')
window.mainloop()
