'''
@Description:
@Author: HuYi
@Date: 2020-05-07 10:37:27
@LastEditors: HuYi
@LastEditTime: 2020-05-07 10:42:01
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


def get_mean_std(dataset, ratio=0.01):
    """Get mean and std by sample ratio
    """
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                             shuffle=True, num_workers=0)
    train = iter(dataloader).next()  # 一个batch的数据
    mean = np.mean(train.numpy(), axis=(0, 2, 3))
    std = np.std(train.numpy(), axis=(0, 2, 3))
    return mean, std


if __name__ == "__main__":

    valset = torchvision.datasets.ImageFolder(
        'datanews/val/', transform=transforms.ToTensor())
    val_mean, val_std = get_mean_std(valset)
    print(val_mean, val_std)
