'''
@Description: 
@Author: HuYi
@Date: 2020-05-12 17:28:19
@LastEditors: HuYi
@LastEditTime: 2020-05-21 16:10:13
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from getStump import get_stump, stump_error

# load data
df_train = pd.read_csv('creditcard_train.csv')
print(df_train.head())
df_train = np.array(df_train)
df_test = pd.read_csv('creditcard_test.csv')
print(df_test.head())
df_test = np.array(df_test)


# make split
X_train = df_train[:, :-2]
y_train = df_train[:, -1]*2-1
X_test = df_test[:, :-2]
y_test = df_test[:, -1]*2-1

# error of AdaBoost
# input
#   X  n*m   sample
#   y  n     label
#   w  n     weight
#   g   1, -1
#   j   dimension
#   a   threshold
#   alpha   weight of classifier
# output
#   error rate of AdaBoost


def AdaBoost_error(X, y, g, a, j, alpha):

    #####################
    ### ADD YOUR CODE ###
    n = X.shape[0]
    numIt = a.shape[0]
    error = 0
    for i in range(n):
        sum = 0
        for k in range(numIt):
            h = (2*(X[i, j[k]] <= a[k])-1)*g[k]
            sum += alpha[k]*h
        error = error+((2*(sum > 0)-1) != y[i])
    error = error/n
    ### ADD YOUR CODE ###
    #####################
    return error


# adaboost
# input
#   X  n*m   sample
#   y  n     label
#   numIt      number of iteration
def AdaBoost(X_train, y_train, X_test, y_test, numIt):
    n = X_train.shape[0]
    w = (1.0/n) * np.ones(n, dtype=np.float)
    g = np.zeros(numIt, dtype=np.float)
    a = np.zeros(numIt, dtype=np.float)
    j = np.zeros(numIt, dtype=np.int)
    alpha = np.zeros(numIt, dtype=np.float)

    train_error = np.zeros(numIt, dtype=np.float)
    test_error = np.zeros(numIt, dtype=np.float)

    for i in range(numIt):
        g[i], a[i], j[i] = get_stump(X_train, y_train, w)
        print(g[i], a[i], j[i])
        e = stump_error(X_train, y_train, w, g[i], a[i], j[i])
        #####################
        ### ADD YOUR CODE ###
        alpha[i] = 0.5*np.log((1-e)/e)
        w = w*np.exp(-alpha[i]*y_train *
                     ((X_train[:, j[i]] <= a[i]) - 0.5) * 2 * g[i])
        w = w/sum(w)
        ### ADD YOUR CODE ###
        #####################
        train_error[i] = AdaBoost_error(X_train, y_train, g, a, j, alpha)
        test_error[i] = AdaBoost_error(X_test, y_test, g, a, j, alpha)

    plt.plot(train_error, 'b')
    plt.plot(test_error, 'y')
    plt.show()


AdaBoost(X_train, y_train, X_test, y_test, 150)
