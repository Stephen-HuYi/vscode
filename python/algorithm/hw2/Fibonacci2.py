'''
@Description: 
@Author: HuYi
@Date: 2020-02-29 18:29:50
@LastEditors: HuYi
@LastEditTime: 2020-03-02 18:01:12
'''
# 方法二：尾递归 O(n)

import time


def fib(n):
    def fib0(n, x, y):
        if n == 0:
            return x
        else:
            return fib0(n-1, y, x+y)
    return fib0(n, 0, 1)


print('Please enter the number of Fibonacci number you want')
i = input()
i = int(i)
start = time.time()
print('F(', i, ')=', fib(i))
end = time.time()
print('totally time cost =', end-start, 'seconds')
# os.system("pause")
