'''
@Description: 
@Author: HuYi
@Date: 2020-04-26 20:17:12
@LastEditors: HuYi
@LastEditTime: 2020-04-26 20:39:22
'''
import sys

if __name__ == "__main__":
    # 读取第一行的n
    n = int(sys.stdin.readline().strip())
    ans = 0
    for i in range(n):
        # 读取每一行
        line = sys.stdin.readline().strip()
        # 把每一行的数字分隔后转化成int列表
        values = list(map(int, line.split()))

    list = []
    while i:
        list.append(i % 2)
        i = i // 2
    list.reverse()
