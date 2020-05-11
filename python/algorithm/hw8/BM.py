'''
@Description: Boyer-Moore
@Author: HuYi
@Date: 2020-05-10 11:32:43
@LastEditors: HuYi
@LastEditTime: 2020-05-10 23:07:09
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


def BM(t, p):
    m = len(p)
    n = len(t)
    if m > n:
        return 'No match!!!'
    skip = []
    for k in range(256):
        skip.append(m)
    for k in range(m - 1):
        skip[ord(p[k])] = m - k - 1
    k = m - 1
    skip = tuple(skip)
    while k < n:
        i, j = k, m - 1
        while j >= 0 and t[i] == p[j]:
            i, j = i-1, j-1
        if j == -1:
            return i + 1
        k += skip[ord(t[k])]
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
    res = BM(t, p)
    end = time.time()
    print('Result:', res)
    print('totally time cost =', end - start, 'seconds')
    # os.system("pause")
