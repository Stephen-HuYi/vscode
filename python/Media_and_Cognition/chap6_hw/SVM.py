'''
@Description:
@Author: HuYi
@Date: 2020-05-11 22:18:26
@LastEditors: HuYi
@LastEditTime: 2020-05-21 17:17:25
'''
from sklearn import svm
import numpy as np
import scipy.io as sio
from sklearn.metrics import precision_score, recall_score

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
print(x_test.shape)
print(y_test.shape)

# 调用SVM，设置参数，请查看SVC的用法
#model = svm.SVC(C=1.0, kernel='linear')
model = svm.SVC(C=1.0, kernel='rbf')
# 学习模型参数
model.fit(x_train, y_train)

# 输出识别准确率
print("SVM-training accuracy:", model.score(x_train, y_train))
y_hat = model.predict(x_train)
# 计算训练集各类别的识别准确率，请根据准确率定义写出计算公式
train_correct = list(0. for i in range(10))
train_total = list(0. for i in range(10))
n = x_train.shape[0]
for i in range(n):
    train_correct[int(y_train[i]-1)] += (y_train[i] == y_hat[i])
    train_total[int(y_train[i]-1)] += 1
for i in range(10):
    print('Accuracy of ', i+1, ':', 100*train_correct[i] / train_total[i])

print("SVM-testing accuracy", model.score(x_test, y_test))
y_hat = model.predict(x_test)
# 计算测试集各类别的识别准确率，同上
test_correct = list(0. for i in range(10))
test_total = list(0. for i in range(10))
n = x_test.shape[0]
for i in range(n):
    test_correct[int(y_test[i]-1)] += (y_test[i] == y_hat[i])
    test_total[int(y_test[i]-1)] += 1
for i in range(10):
    print('Accuracy of ', i+1, ':', 100*test_correct[i] / test_total[i])
