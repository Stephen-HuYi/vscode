'''
@Description:
@Author: HuYi
@Date: 2020-03-08 11:24:32
@LastEditors: HuYi
@LastEditTime: 2020-03-10 22:42:46
'''
import random
import math
import time
import tkinter as tk


def get_dis(array):
    return math.sqrt((array[0][0]-array[1][0])**2+(array[0][1]-array[1][1])**2)


# 暴力算法
def brute(array):
    dis_min = float('inf')
    pair = []
    for i in range(len(array)):
        for j in range(i+1, len(array)):
            dis = get_dis([array[i], array[j]])
            if dis < dis_min:
                dis_min = dis
                pair = [array[i], array[j]]
    return [pair, dis_min]


# 图形界面实现
window = tk.Tk()
window.title('window')
window.geometry('800x600')
show = tk.Message(window, text="计算结果(Θ(n^2)算法)", font=('Arial', 12), width=200)
show.pack()
text = tk.Text(window, width=60, height=6)
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
    array = [(random.randint(-10000, 10000), random.randint(-10000, 10000))
             for x in range(num)]
    start = time.time()
    temp = brute(array)
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
    temp = brute(array)
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
