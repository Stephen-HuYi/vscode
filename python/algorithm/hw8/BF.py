'''
@Description:Brute-Force 
@Author: HuYi
@Date: 2020-05-10 11:32:50
@LastEditors: HuYi
@LastEditTime: 2020-05-10 23:09:46
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


def BF(t, p):
    m, n = len(t), len(p)
    if m < n:
        return 'No match!!!'
    i, j = 0, 0
    while i < m and j < n:
        if list(t)[i] == list(p)[j]:
            i, j = i+1, j+1
        else:
            i = i-j+1
            j = 0
            if(i > m-n):
                return 'No match!!!'
    if j >= n:
        return i-n
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
    res = BF(t, p)
    end = time.time()
    print('Result:', res)
    print('totally time cost =', end - start, 'seconds')
    os.system("pause")
