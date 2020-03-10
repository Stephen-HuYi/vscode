'''
@Description:
@Author: HuYi
@Date: 2020-03-08 11:24:32
@LastEditors: HuYi
@LastEditTime: 2020-03-10 14:06:32
'''
import random
import math
import tkinter as tk


def calDis(seq):
    dis = math.sqrt((seq[0][0]-seq[1][0])**2+(seq[0][1]-seq[1][1])**2)
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


def combine(left, right, resMin, med_x):
    dis = resMin[1]
    minDis = resMin[1]
    pair = resMin[0]
    for u in left:
        if u[0] < med_x-dis:
            continue
        for v in candidateDot(u, right, dis, med_x):
            dis = calDis([u, v])
            if dis < minDis:
                minDis = dis
                pair = [u, v]
    return [pair, minDis]


# 分治求解
def divide(seq):
    # 求序列元素数量
    n = len(seq)
    # 按点的纵坐标升序排序
    seq = sorted(seq)
    # 递归开始进行
    if n <= 1:
        return None, float('inf')
    elif n == 2:
        return [seq, calDis(seq)]
    else:
        half = int(len(seq)/2)
        med_x = (seq[half][0]+seq[-half-1][0])/2
        left = seq[:half]
        resLeft = divide(left)
        right = seq[half:]
        resRight = divide(right)
        # 获取两集合中距离最短的点对
        if resLeft[1] < resRight[1]:
            resMin = combine(left, right, resLeft, med_x)
        else:
            resMin = combine(left, right, resRight, med_x)
        pair = resMin[0]
        minDis = resMin[1]
    return [pair, minDis]


window = tk.Tk()
window.title('window')
window.geometry('800x600')
text = tk.Text(window, width=40, height=8)
text.pack()
e = tk.Entry(window, show=None)
e.pack()
w = tk.Canvas(window, width=300, height=300, background='white')
w.pack()
c = tk.Button(window, text='calc', font=('Arial', 12),
              width=10, height=1, command=lambda: calc(array))
c.pack()


def handle(event, array):
    array.append((event.x, event.y))
    x1, y1 = (event.x - 1), (event.y - 1)
    x2, y2 = (event.x + 1), (event.y + 1)
    w.create_oval(x1, y1, x2, y2, fill='red')


def moderandom():
    num = int(e.get())
    array = [(random.randint(-10000, 10000), random.randint(-10000, 10000))
             for x in range(num)]
    temp = divide(array)
    text.delete(1.0, tk.END)
    text.insert(1.0, "The result of moderandom:\n")
    text.insert(2.0, "points:\n")
    text.insert(3.0, temp[0])
    text.insert(4.0, "\n")
    text.insert(4.0, "distance:\n")
    text.insert(5.0, temp[1])


def calc(array):
    text.delete(1.0, tk.END)
    temp = divide(array)
    text.insert(1.0, "The result of modeclick:\n")
    text.insert(2.0, "points:\n")
    text.insert(3.0, temp[0])
    text.insert(4.0, "\n")
    text.insert(4.0, "distance:\n")
    text.insert(5.0, temp[1])


def modeclick():
    array = []
    w.bind("<Button-1>", lambda event: handle(event, array))
    w.bind("<Button-2>", lambda event: handle(event, array))
    w.bind("<Button-3>", lambda event: handle(event, array))


a = tk.Button(window, text='moderandom', font=('Arial', 12),
              width=10, height=1, command=moderandom)
a.pack()
b = tk.Button(window, text='modeclick', font=('Arial', 12),
              width=10, height=1, command=modeclick)
b.pack()
window.mainloop()
