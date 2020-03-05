'''
@Description: 
@Author: HuYi
@Date: 2020-02-29 18:29:50
@LastEditors: HuYi
@LastEditTime: 2020-03-05 11:22:05
'''
# 方法一：递归 O(1.618^n)


import time


def fib(n):
    if n < 0:
        return 'illegal input!!!'
    if n <= 1:
        return n
    return fib(n-1) + fib(n-2)


print('Please enter the number of Fibonacci number you want')
i = input()
i = int(i)
start = time.time()
print('F(', i, ')=', fib(i))
end = time.time()
print('totally time cost =', end-start, 'seconds')
# os.system("pause")
