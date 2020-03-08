'''
@Description: 
@Author: HuYi
@Date: 2020-03-08 14:57:03
@LastEditors: HuYi
@LastEditTime: 2020-03-08 16:35:16
'''
import tkinter as tk


def callback(event):
    print("当前位置：", event.x, event.y)


window = tk.Tk()
window.title('window')
window.geometry('500x300')
# 第4步，在图形界面上创建一个标签用以显示内容并放置
tk.Label(window, text='on the window', bg='red', font=(
    'Arial', 16)).pack()   # 和前面部件分开创建和放置不同，其实可以创建和放置一步完成

# 第5步，创建一个主frame，长在主window窗口上
frame = tk.Frame(window, width=300, height=300)
frame.bind("<Button-1>", callback)
frame.bind("<Button-2>", callback)
frame.bind("<Button-3>", callback)
frame.pack()


window.mainloop()
