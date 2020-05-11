'''
@Description: 
@Author: HuYi
@Date: 2020-05-10 11:32:38
@LastEditors: HuYi
@LastEditTime: 2020-05-10 23:08:59
'''


import string
import random
import time


def ranstr(filename, num):
    H = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789,./?<>;:!@#$%^&*()+-='
    txt = ''
    for i in range(num):
        txt += random.choice(H)
    with open(filename, "a") as f:
        f.write(txt)


def Next(p):
    i, j = 0, -1
    next = [-1]
    while(i < len(p)-1):
        if(j == -1 or p[i] == p[j]):
            i, j = i+1, j+1
            next.append(j)
        else:
            j = next[j]
    return next


def KMP(t, p, pos=0):
    i, j = pos, 0
    next = Next(p)
    while(i < len(t) and j < len(p)):
        if(j == -1 or t[i] == p[j]):
            i, j = i+1, j+1
        else:
            j = next[j]
    if(j >= len(p)):
        return i - len(p)  # 说明匹配到最后了
    else:
        return 'No match!!!'


if __name__ == "__main__":
    print('TXT:Randomly generated(input0) or existing files(input1)?')
    i = int(input())
    if (i == 0):
        print('Please enter a file name')
        filename = str(input())
        print('Please enter a string length')
        strlen = int(input())
        ranstr(filename, strlen)
    elif (i == 1):
        print('Please enter a file name')
        filename = str(input())
    else:
        print('invalid input!!!')
    with open(filename, "r") as f:  # 打开文件
        t = f.read()  # 读取文件
    print('Please enter the pattern')
    p = str(input())
    start = time.time()
    res = KMP(t, p)
    end = time.time()
    print('Result:', res)
    print('totally time cost =', end - start, 'seconds')
    # os.system("pause")
