'''
@Description: 
@Author: HuYi
@Date: 2020-02-29 18:46:03
@LastEditors: HuYi
@LastEditTime: 2020-03-02 19:57:22
'''
# 方法三：循环迭代 O(n)


import time


def fib(n):
    a, b = 0,  1
    if n < 0:
        return 'illegal input!!!'
    else:
        for k in range(n):
            a, b = b, a + b
        return a


print('Please enter the number of Fibonacci number you want')
i = input()
i = int(i)
start = time.time()
print('F(', i, ')=', fib(i))
end = time.time()
print('totally time cost =', end-start, 'seconds')
# os.system("pause")
