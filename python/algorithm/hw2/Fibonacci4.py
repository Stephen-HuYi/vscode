'''
@Description: 
@Author: HuYi
@Date: 2020-03-01 14:58:54
@LastEditors: HuYi
@LastEditTime: 2020-03-02 18:02:45
'''
# 方法四：矩阵快速幂 O(lgn)


import time


def fib(n):
    F = [1, 1, 1, 0]
    if n < 0:
        return 'illegal input!!!'
    elif n == 0:
        return 0
    elif n >= 1:
        fast_pow(F, n-1)
        return F[0]


# 2 x 2 矩阵快速幂

def fast_pow(mat, n):
    if n == 1 or n == 0:
        pass
    elif n == 2:
        mat1 = [mat[0], mat[1], mat[2], mat[3]]
        mat2 = [mat[0], mat[1], mat[2], mat[3]]
        for i in range(2):
            for j in range(2):
                mat[2*i + j] = cal(mat1[i::2], mat2[j::2])
    elif n % 2 == 0:
        fast_pow(mat, n/2)
        fast_pow(mat, 2)
    else:
        mat3 = [mat[0], mat[1], mat[2], mat[3]]
        fast_pow(mat, n-1)
        mat4 = [mat[0], mat[1], mat[2], mat[3]]
        for i in range(2):
            for j in range(2):
                mat[2*i + j] = cal(mat3[i::2], mat4[j::2])
    return mat


def cal(row, col):
    ans = 0
    for i in range(2):
        ans += row[i] * col[i]
    return ans


print('Please enter the number of Fibonacci number you want')
i = input()
i = int(i)
start = time.time()
print('F(', i, ')=', fib(i))
end = time.time()
print('totally time cost =', end-start, 'seconds')
# os.system("pause")
