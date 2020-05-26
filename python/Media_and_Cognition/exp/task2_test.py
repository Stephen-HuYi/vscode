'''
@Description:
@Author: HuYi
@Date: 2020-05-22 21:12:35
@LastEditors: HuYi
@LastEditTime: 2020-05-22 21:42:27
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
from torchvision.datasets import ImageFolder
root = 'D:/vscode/python/Media_and_Cognition/exp/'


class Myfolder(ImageFolder):

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img, fn, label


if __name__ == "__main__":
    # 预处理
    transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(
    ), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    testset = Myfolder(root+'Test/', transform=transform)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=1, shuffle=True, num_workers=0)
    classes = ('i2', 'i4', 'i5', 'io', 'ip', 'p11', 'p23', 'p26', 'p5', 'pl30',
               'pl40',  'pl5', 'pl50', 'pl60', 'pl80', 'pn', 'pne', 'po', 'w57')

    mydict = {}
    for data in testloader:
        images, fn, labels = data
        print(str(fn)[-11:-3])
        print(classes[labels])
        mydict[str(fn)[-11:-3]] = str(classes[labels])
    with open(root+'predhhh.json', 'w', encoding='utf-8') as f:
        json.dump(mydict, f)
