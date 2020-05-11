'''
@Description:
@Author: HuYi
@Date: 2020-05-06 11:59:42
@LastEditors: HuYi
@LastEditTime: 2020-05-11 22:36:02
'''
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision
import torch.nn.functional as F
import os
import numpy as np
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch
import json
root = 'D:/vscode/python/Media_and_Cognition/exp/'


def default_loader(path):
    return Image.open(root+'datanews/test/'+path).convert('RGB')


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


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


if __name__ == "__main__":
    # 预处理
    transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(
    ), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    testset = MyDataset(js=root+'datanews/test.json', transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=1, shuffle=False, num_workers=0)
    classes = ('i2', 'i4', 'i5', 'io', 'ip', 'p11', 'p23', 'p26', 'p5', 'pl30',
               'pl40',  'pl5', 'pl50', 'pl60', 'pl80', 'pn', 'pne', 'po', 'w57')
    PATH = root+'CNN_best.pth'
    net = Net()
    net.load_state_dict(torch.load(PATH))
    mydict = {}
    with torch.no_grad():
        for data in testloader:
            images, fn, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            mydict[str(fn)[2:-3]] = str(classes[predicted])
    with open(root+'pred.json', 'w', encoding='utf-8') as f:
        json.dump(mydict, f)
