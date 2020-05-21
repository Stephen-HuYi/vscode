'''
@Description:
@Author: HuYi
@Date: 2020-05-12 17:28:19
@LastEditors: HuYi
@LastEditTime: 2020-05-21 15:51:36
'''
import numpy as np

# make stump
# input
#   X  n*m   sample
#   y  n     label
#   w  n     weight
# output
#   g   1, -1
#   j   dimension
#   a   threshold
# h(x) = g when x[j] <= a, else h(x) = -g


def get_stump(X, y, w):
    #####################
    ### ADD YOUR CODE ###
    n, m = X.shape
    num = 100
    e_min = float('inf')
    for j in range(m):
        max_data, min_data = max(X[:, j]), min(X[:, j])
        step_size = (max_data-min_data)/num
        for k in range(num+1):
            a = min_data + k*step_size
            e1, e2 = stump_error(X, y, w, 1, a, j), stump_error(
                X, y, w, -1, a, j)
            if e1 < e2:
                e, g = e1, 1
            else:
                e, g = e2, -1
            if e < e_min:
                e_min, g_final, a_final, j_final = e, g, a, j
    ### ADD YOUR CODE ###
    #####################
    return g_final, a_final, j_final


# error of stump
# input
#   X  n*m   sample
#   y  n     label
#   w  n     weight
#   g   1, -1
#   j   dimension
#   a   threshold
# output
#   error
#   error of stump
def stump_error(X, y, w, g, a, j):
    p = ((X[:, j] <= a) - 0.5) * 2 * g
    e = sum((p != y) * w)
    return e
