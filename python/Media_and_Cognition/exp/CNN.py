'''
@Description:
@Author: HuYi
@Date: 2020-05-03 15:58:38
@LastEditors: HuYi
@LastEditTime: 2020-05-11 22:35:40
'''
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
root = 'D:/vscode/python/Media_and_Cognition/exp/'

# functions to show an image


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
    trainset = torchvision.datasets.ImageFolder(
        root+'datanews/train/', transform=transform)
    print(len(trainset))
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=4, shuffle=True, num_workers=0)
    # print(trainset.imgs)
    valset = torchvision.datasets.ImageFolder(
        root+'datanews/val/', transform=transform)
    print(len(valset))
    valloader = torch.utils.data.DataLoader(
        valset, batch_size=1, shuffle=True, num_workers=0)
    # 利用自定义dateset读取测试数据对象，并设定batch-size和工作现场
    classes = ('i2', 'i4', 'i5', 'io', 'ip', 'p11', 'p23', 'p26', 'p5', 'pl30',
               'pl40',  'pl5', 'pl50', 'pl60', 'pl80', 'pn', 'pne', 'po', 'w57')
    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    net = Net()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    PATH_vec = ['CNN1.pth', 'CNN2.pth', 'CNN3.pth', 'CNN4.pth', 'CNN5.pth', 'CNN6.pth', 'CNN7.pth', 'CNN8.pth', 'CNN9.pth', 'CNN10.pth',
                'CNN11.pth', 'CNN12.pth', 'CNN13.pth', 'CNN14.pth', 'CNN15.pth', 'CNN16.pth', 'CNN17.pth', 'CNN18.pth', 'CNN19.pth', 'CNN20.pth', ]

    loss_vec = [0 for _ in range(20)]
    for epoch in range(20):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
        #print('Finished Training')
        loss_vec[epoch] = running_loss
        print('[%d, %5d] loss: %.3f' %
              (epoch + 1, i + 1, loss_vec[epoch] / len(trainset)))
        PATH = root+PATH_vec[epoch]
        torch.save(net.state_dict(), PATH)

    dataiter = iter(valloader)
    images, labels = dataiter.next()

    # print images
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' %
                                    classes[labels[j]] for j in range(1)))
    net = Net()
    min_index, min_number = min(
        enumerate(loss_vec), key=operator.itemgetter(1))
    PATH = PATH_vec[min_index]
    net.load_state_dict(torch.load(PATH))
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join('%5s' %
                                  classes[predicted[j]] for j in range(1)))
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
