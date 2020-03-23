'''
@Description: 
@Author: HuYi
@Date: 2020-03-23 19:35:57
@LastEditors: HuYi
@LastEditTime: 2020-03-23 19:42:47
'''
import random
import time


def insert_sort(a):
    n = len(a)
    for j in range(0, n):
        for i in range(j, 0, -1):
            if a[i] < a[i - 1]:
                a[i], a[i - 1] = a[i - 1], a[i]
            else:
                break
    return a


print('Please enter the number of Fibonacci number you want')
i = input()
i = int(i)
a = []
for k in range(i):
    a.append(rand() << 16 | rand())
start = time.time()
insert_sort(a)
end = time.time()
print('totally time cost =', end-start, 'seconds')
# os.system("pause")
