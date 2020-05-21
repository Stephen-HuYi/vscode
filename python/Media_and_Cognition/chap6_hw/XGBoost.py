'''
@Description: 
@Author: HuYi
@Date: 2020-05-21 15:46:59
@LastEditors: HuYi
@LastEditTime: 2020-05-21 16:35:19
'''


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
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
y_test = df_test[:, -1] * 2 - 1

numIt = 150
train_error = np.zeros(numIt, dtype=np.float)
test_error = np.zeros(numIt, dtype=np.float)


for i in range(numIt):
    num_round = i+1
    bst = XGBClassifier(max_depth=1, learning_rate=0.5,
                        n_estimators=num_round, objective='binary:logistic')
    eval_set = [(X_train, y_train), (X_test, y_test)]
    bst.fit(X_train, y_train, eval_set=eval_set)
    train_preds = bst.predict(X_train)
    train_error[i] = 1-accuracy_score(y_train, train_preds)
    preds = bst.predict(X_test)
    test_error[i] = 1-accuracy_score(y_test, preds)


plt.plot(train_error, 'b')
plt.plot(test_error, 'y')
plt.show()
