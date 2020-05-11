'''
@Description: 
@Author: HuYi
@Date: 2020-05-06 19:23:57
@LastEditors: HuYi
@LastEditTime: 2020-05-07 15:23:38
'''
'''
@Description:
@Author: HuYi
@Date: 2020-05-03 15:58:38
@LastEditors: HuYi
@LastEditTime: 2020-05-06 19:00:50
'''


# functions to show an image




import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
import os
import operator
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 19)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    # 预处理
    transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(
    ), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # 使用torchvision.datasets.ImageFolder读取数据集 指定train 和 val文件夹
    # print(trainset.imgs)
    valset = torchvision.datasets.ImageFolder(
        'datanews/val/', transform=transform)
    print(len(valset))
    valloader = torch.utils.data.DataLoader(
        valset, batch_size=1, shuffle=False, num_workers=0)
    # 利用自定义dateset读取测试数据对象，并设定batch-size和工作现场
    classes = ('i2', 'i4', 'i5', 'io', 'ip', 'p11', 'p23', 'p26', 'p5', 'pl30',
               'pl40',  'pl5', 'pl50', 'pl60', 'pl80', 'pn', 'pne', 'po', 'w57')
    # get some random training images
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    dataiter = iter(valloader)
    images, labels = dataiter.next()

    PATH = './CNN_best.pth'
    net.load_state_dict(torch.load(PATH))
    correct = 0
    total = 0
    with torch.no_grad():
        for data in valloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the val images: %.3f %%' % (
        100*correct / total))
    class_correct = list(0. for i in range(19))
    class_total = list(0. for i in range(19))
    with torch.no_grad():
        for data in valloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(1):
                label = labels[i]
                class_correct[label] += c.item()
                class_total[label] += 1

    for i in range(19):
        print('Accuracy of %5s : %.3f %%' % (
            classes[i], 100*class_correct[i] / class_total[i]))
