'''
@Description:
@Author: HuYi
@Date: 2020-04-04 23:15:39
@LastEditors: HuYi
@LastEditTime: 2020-04-06 22:44:46
'''
import math


def bin_search(last, cnt, w):
    left = 0
    right = cnt
    while (left <= right):
        mid = int((left + right) / 2)
        if (last[mid] > w):
            right = mid - 1
        elif (last[mid] < w):
            left = mid + 1
        else:
            return mid
    return left


def cal(a):
    N = len(a)
    cnt = 0
    last = []
    last.append(a[0])
    for i in range(1, N):
        if (a[i] > last[cnt]):
            cnt += 1
            last.append(a[i])
        else:
            location = bin_search(last, cnt, a[i])
            last[location] = a[i]
    length = cnt+1
    last.append(math.inf)
    lis = [0]*length
    for i in range(N-1, -1, -1):
        if (last[cnt] <= a[i] < last[cnt+1]):
            lis[cnt] = a[i]
            cnt -= 1
        if (cnt == -1):
            break
    print('The length of LIS is:', length)
    print('LIS:', lis)


print('Please enter the sequence, separated by only one space, ended by enter')
a = input()
n = (len(a)+1)/2
array = [int(n) for n in a.split()]
cal(array)
os.system("pause")
