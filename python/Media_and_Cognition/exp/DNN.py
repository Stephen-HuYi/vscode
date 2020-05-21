import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import random
import itertools
import time
from PIL import Image
import numpy as np
import cv2
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
import os
import operator
from utils import *


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2_1 = nn.Conv2d(6, 16, 5)
        self.fc1_1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2_1 = nn.Linear(120, 84)
        self.fc3_1 = nn.Linear(84, 19)

    def forward(self, x):
        x = x.float()
        x1 = self.pool(F.relu(self.conv1_1(x)))
        x1 = self.pool(F.relu(self.conv2_1(x1)))
        x1 = x1.view(-1, 16 * 5 * 5)
        x1 = F.relu(self.fc1_1(x1))
        x1 = F.relu(self.fc2_1(x1))
        x1 = self.fc3_1(x1)
        return x1


class Model:

    def __init__(self, train_path, classes, config):
        print('Init started.')
        self.classes = classes
        self.label_dict = {label:i for i,label in enumerate(classes)}   # label->ids
        self.learning_rate = config['lr']
        self.momentum = config['momentum']
        self.batch_size = config['batch_size']
        self.epoch_num = config['epoch_num']
        self.valid_epochs = config['validation_epochs']
        self.data = load_images(train_path, labels=classes)
        self.train_data = self.valid_data = []
        self.split_set()
        self.transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(
        ), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        # 使用torchvision.datasets.ImageFolder读取数据集 指定train 和 val文件夹
        self.trainset = torchvision.datasets.ImageFolder(
            'datanews/train/', transform=self.transform)
        self.trainloader = torch.utils.data.DataLoader(
            self.trainset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        # print(trainset.imgs)
        self.valset = torchvision.datasets.ImageFolder(
            'datanews/valid/', transform=self.transform)
        self.valloader = torch.utils.data.DataLoader(
            self.valset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        # 利用自定义dateset读取测试数据对象，并设定batch-size和工作现场
        print('Split ended.')
        self.net1 = Net()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(itertools.chain(self.net1.parameters()), lr=self.learning_rate, momentum=self.momentum)
        print('Init ended.')

    def split_set(self):
        for label in self.classes:
            cnt = 0
            tmp_list = self.data[label]
            random.shuffle(tmp_list)
            for img in tmp_list:
                if cnt < round(len(tmp_list)*0.8):
                    self.train_data.append((img, self.label_dict[label]))
                else:
                    self.valid_data.append((img, self.label_dict[label]))
                cnt += 1

    def evaluate(self, is_train):
        dataset = self.train_data if is_train else self.valid_data
        correct = 0
        total = 0
        running_loss = 0.0
        batch_num = int(np.ceil(len(dataset)/self.batch_size))
        print('{} pics, batch_size {}, batch_num {}'.format(len(dataset),self.batch_size,batch_num))
        for i, data in enumerate(self.trainloader, 0):
            #tmp_data = dataset[int(i)*self.batch_size:int(i+1)*self.batch_size]
            #print(len(tmp_data))
            #if len(tmp_data) == 0:
                #break
            #input_x1  = np.zeros((len(tmp_data), 3, 32, 32))
            #input_y = []
            #cnt = 0
            #for img, label in tmp_data:
                #print(img.shape)
                #print(label)
                #img1 = img
                #if is_train:
                    #img1 = self.distortion(img1)
                #img1 = cv2.resize(img1, (32, 32))
                #cv2.imshow('img', img1)
                #cv2.waitKey(0)
                #img1 = img1.astype(np.float32)
                #img1 = ((img1 / img1.max()) -0.5)/0.5
                #for r in range(3):
                    #input_x1[cnt, r, :, :] = img1[:, :, r]
                #input_y.append(label)
                #cnt += 1
            #cv2.imshow('img', input_x1[0,0,:,:])
            #cv2.waitKey(0)
            #print(input_x2.shape)

            #input_y = np.asarray(input_y)
            inputs, labels = data
            if is_train:
                self.optimizer.zero_grad()
                #input_x1 = torch.from_numpy(input_x1)
                #input_y = torch.from_numpy(input_y)
                output = self.net1(inputs)
                #output = output.long()
                #input_y = input_y.long()
                #print(input_y.shape)
                #print(output.shape)
                loss = self.criterion(output, labels)
                _, predicted = torch.max(output, 1)
                loss.backward()
                self.optimizer.step()
            else:
                with torch.no_grad():
                    #input_x1 = torch.from_numpy(input_x1)
                    #input_y = torch.from_numpy(input_y)
                    output = self.net1(inputs)
                    #input_y = input_y.long()
                    loss = self.criterion(output, labels)
                    _, predicted = torch.max(output, 1)
            running_loss += loss.item()
            #total += input_y.size(0)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            print('Current avg loss {}'.format(loss.item()/labels.size(0)))
        return correct, total, running_loss

    @staticmethod
    def distortion(img):
        img_size = np.shape(img)
        resize_ratio = np.random.uniform(0.9, 1.1)
        yshift = np.random.uniform(-0.1 * img_size[0], 0.1 * img_size[0])
        xshift = np.random.uniform(-0.1 * img_size[1], 0.1 * img_size[1])
        M = np.float32([[1, 0, round(xshift)], [0, 1, round(yshift)]])
        img = cv2.warpAffine(img, M, (img_size[1], img_size[0]))
        img = cv2.resize(img, None, fx=resize_ratio, fy=resize_ratio)
        img_size = np.shape(img)
        rotate_ratio = np.random.uniform(-5, 5)
        M = cv2.getRotationMatrix2D((img_size[1] / 2, img_size[0] / 2), rotate_ratio, 1)
        img = cv2.warpAffine(img, M, (img_size[1], img_size[0]))
        return img


    def train(self):
        #todo: training process
        print('===========Training started===============')
        best_acc = 0.0
        log_file = open('./train_log_{}.txt'.format(time.strftime("%d-%H-%M")),'w',encoding='utf8')
        for i in range(self.epoch_num):
            correct, total, loss = self.evaluate(is_train=True)
            print('epoch '+str(int(i))+'finished, acc {}, loss {}'.format(correct/total, loss/len(self.train_data)))
            log_file.write('{}\tepoch'.format(time.strftime("%d-%H-%M"))+str(int(i))+'finished, acc {}, loss {}'.format(correct/total, loss/len(self.train_data)))
            log_file.flush()
            state = {'net1':self.net1.state_dict(), 'optimizer':self.optimizer.state_dict(), 'epoch':i}
            torch.save(state, './MCDNN.pth')
            if (i+1) % self.valid_epochs == 0:
                correct, total, loss = self.evaluate(is_train=False)
                print('Validation ended, acc {}, loss {}'.format(correct / total, loss/len(self.valid_data)))
                log_file.write('{}\tValidation ended, acc {}, loss {}'.format(time.strftime("%d-%H-%M"), correct / total, loss/len(self.valid_data)))
                log_file.flush()
                if (i+1) == self.valid_epochs:
                    best_acc = correct/total
                    print('Best model saved.')
                    log_file.write('Best model saved.')
                    log_file.flush()
                    torch.save(state, './MCDNN_best.pth')
                else:
                    acc = correct/total
                    if acc > best_acc:
                        best_acc = acc
                        print('Best model saved.')
                        log_file.write('Best model saved.')
                        log_file.flush()
                        torch.save(state, './MCDNN_best.pth')
        log_file.close()





train_path = './Train/'
classes = ('i2', 'i4', 'i5', 'io', 'ip', 'p11', 'p23', 'p26', 'p5', 'pl30',
               'pl40',  'pl5', 'pl50', 'pl60', 'pl80', 'pn', 'pne', 'po', 'w57')
config_dict = {'batch_size':4,
               'lr':0.001,
               'momentum':0.9,
               'epoch_num':20,
               'validation_epochs':5}

model = Model(train_path=train_path, classes=classes, config=config_dict)
model.train()

