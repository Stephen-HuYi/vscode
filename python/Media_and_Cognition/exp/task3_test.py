'''
@Description: 
@Author: HuYi
@Date: 2020-05-08 13:29:03
@LastEditors: HuYi
@LastEditTime: 2020-05-08 13:46:43
'''
'''
@Description:
@Author: HuYi
@Date: 2020-05-08 12:17:26
@LastEditors: HuYi
@LastEditTime: 2020-05-08 13:25:35
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
from torchvision import models
import os
import operator
import json
from PIL import Image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def default_loader(path):
    return Image.open('D:/vscode/python/DataFewShot/test/'+path).convert('RGB')


class MyDataset(Dataset):
    def __init__(self, js, transform=None, target_transform=None, loader=default_loader):
        test_annotation = json.load(open(js))
        imgs = []
        for k, v in test_annotation.items():
            imgs.append((k, v))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img, fn, label

    def __len__(self):
        return len(self.imgs)


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


class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 11)

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
    testset = MyDataset(js='D:/vscode/python/DataFewShot/test.json',
                        transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=1, shuffle=False, num_workers=0)
    # 利用自定义dateset读取测试数据对象，并设定batch-size和工作现场
    classes = ('p1', 'p12', 'p14', 'p17', 'p19',
               'p22', 'p25', 'p27', 'p3', 'p6', 'p9')

    net = Net2()
    PATH = './finetune20.pth'
    net.load_state_dict(torch.load(PATH))
    mydict = {}
    with torch.no_grad():
        for data in testloader:
            images, fn, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)

            print(fn, predicted, classes[predicted])
            imshow(torchvision.utils.make_grid(images))
            mydict[str(fn)[2:-3]] = str(classes[predicted])
