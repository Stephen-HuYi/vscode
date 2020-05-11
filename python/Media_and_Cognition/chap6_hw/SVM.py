'''
@Description: 
@Author: HuYi
@Date: 2020-05-11 22:18:26
@LastEditors: HuYi
@LastEditTime: 2020-05-11 22:22:09
'''
from sklearn import svm
import numpy as np
import scipy.io as sio

# 装载数据
data = sio.loadmat(
    'D:/vscode/python/Media_and_Cognition/chap6_hw/Caltech-256_VGG_10classes.mat')
traindata = data['traindata']
testdata = data['testdata']

x_train = traindata[0][0][0].transpose()
y_train = traindata[0][0][1].ravel()
x_test = testdata[0][0][0].transpose()
y_test = testdata[0][0][1].ravel()

# check if the data have been correctly loaded
print(x_train.shape)
print(y_train.shape)

# 调用SVM，设置参数，请查看SVC的用法
model = svm.SVC()
# 学习模型参数
model.fit(x_train, y_train)

# 输出识别准确率
print("SVM-training accuracy:", model.score(x_train, y_train))
y_hat = model.predict(x_train)
# 计算训练集各类别的识别准确率，请根据准确率定义写出计算公式

print("SVM-testing accuracy", model.score(x_test, y_test))
y_hat = model.predict(x_test)
# 计算测试集各类别的识别准确率，同上
