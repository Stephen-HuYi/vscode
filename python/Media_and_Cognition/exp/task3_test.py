'''
@Description: 
@Author: HuYi
@Date: 2020-05-23 12:21:57
@LastEditors: HuYi
@LastEditTime: 2020-05-24 16:49:52
'''
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import random
import itertools
import time
import json
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
        # DNN1
        self.conv1_1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2_1 = nn.Conv2d(6, 16, 5)
        self.fc1_1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2_1 = nn.Linear(120, 84)
        self.fc3_1 = nn.Linear(84, 11)
        # DNN2
        self.conv1_2 = nn.Conv2d(3, 6, 5)
        self.conv2_2 = nn.Conv2d(6, 16, 5)
        self.fc1_2 = nn.Linear(16 * 5 * 5, 120)
        self.fc2_2 = nn.Linear(120, 84)
        self.fc3_2 = nn.Linear(84, 11)
        # DNN3
        self.conv1_3 = nn.Conv2d(3, 6, 5)
        self.conv2_3 = nn.Conv2d(6, 16, 5)
        self.fc1_3 = nn.Linear(16 * 5 * 5, 120)
        self.fc2_3 = nn.Linear(120, 84)
        self.fc3_3 = nn.Linear(84, 11)
        # DNN4
        self.conv1_4 = nn.Conv2d(3, 6, 5)
        self.conv2_4 = nn.Conv2d(6, 16, 5)
        self.fc1_4 = nn.Linear(16 * 5 * 5, 120)
        self.fc2_4 = nn.Linear(120, 84)
        self.fc3_4 = nn.Linear(84, 11)
        # DNN5
        self.conv1_5 = nn.Conv2d(3, 6, 5)
        self.conv2_5 = nn.Conv2d(6, 16, 5)
        self.fc1_5 = nn.Linear(16 * 5 * 5, 120)
        self.fc2_5 = nn.Linear(120, 84)
        self.fc3_5 = nn.Linear(84, 11)

    def forward(self, x):
        x = x.float()
        x1 = self.pool(F.relu(self.conv1_1(x)))
        x1 = self.pool(F.relu(self.conv2_1(x1)))
        x1 = x1.view(-1, 16 * 5 * 5)
        x1 = F.relu(self.fc1_1(x1))
        x1 = F.relu(self.fc2_1(x1))
        x1 = self.fc3_1(x1)
        x2 = self.pool(F.relu(self.conv1_2(x)))
        x2 = self.pool(F.relu(self.conv2_2(x2)))
        x2 = x2.view(-1, 16 * 5 * 5)
        x2 = F.relu(self.fc1_2(x2))
        x2 = F.relu(self.fc2_2(x2))
        x2 = self.fc3_2(x2)
        x3 = self.pool(F.relu(self.conv1_3(x)))
        x3 = self.pool(F.relu(self.conv2_3(x3)))
        x3 = x3.view(-1, 16 * 5 * 5)
        x3 = F.relu(self.fc1_3(x3))
        x3 = F.relu(self.fc2_3(x3))
        x3 = self.fc3_3(x3)
        x4 = self.pool(F.relu(self.conv1_4(x)))
        x4 = self.pool(F.relu(self.conv2_4(x4)))
        x4 = x4.view(-1, 16 * 5 * 5)
        x4 = F.relu(self.fc1_4(x4))
        x4 = F.relu(self.fc2_4(x4))
        x4 = self.fc3_4(x4)
        x5 = self.pool(F.relu(self.conv1_5(x)))
        x5 = self.pool(F.relu(self.conv2_5(x5)))
        x5 = x5.view(-1, 16 * 5 * 5)
        x5 = F.relu(self.fc1_5(x5))
        x5 = F.relu(self.fc2_5(x5))
        x5 = self.fc3_5(x5)
        return x1+x2+x3+x4+x5


class Model:

    def __init__(self, data_path, classes, config):
        torch.cuda.set_device(0)
        print('Init started.')
        self.classes = classes
        self.is_train = config['is_train']
        self.label_dict = {label: i for i,
                           label in enumerate(classes)}   # label->ids
        self.label_dict_reverse = {v: i for i, v in self.label_dict.items()}
        self.learning_rate = config['lr']
        #self.momentum = config['momentum']
        self.batch_size = config['batch_size']
        self.epoch_num = config['epoch_num']
        self.valid_epochs = config['validation_epochs']
        self.epoch = 0
        self.gpus = [0]
        self.cuda_gpu = torch.cuda.is_available()
        self.data, self.data_name = load_images(
            data_path, labels=classes, is_train=self.is_train)
        if self.is_train:
            self.train_data = self.valid_data = []
            self.split_set()
            random.shuffle(self.train_data)
            print('Split and shuffle ended.')
        else:
            self.test_data = self.data
            self.pred = []
            print('Test dataset generated.')
        self.net1 = Net()
        self.net2 = Net()
        self.net3 = Net()
        self.net4 = Net()
        self.net5 = Net()
        if self.cuda_gpu:
            self.net1 = torch.nn.DataParallel(
                self.net1, device_ids=self.gpus).cuda()
            self.net2 = torch.nn.DataParallel(
                self.net2, device_ids=self.gpus).cuda()
            self.net3 = torch.nn.DataParallel(
                self.net3, device_ids=self.gpus).cuda()
            self.net4 = torch.nn.DataParallel(
                self.net4, device_ids=self.gpus).cuda()
            self.net5 = torch.nn.DataParallel(
                self.net5, device_ids=self.gpus).cuda()
        self.criterion = nn.CrossEntropyLoss()
        ignored_params = list(map(id,
                                  list(self.net1.module.fc3_1.parameters()) + list(self.net1.module.fc3_2.parameters()) + list(self.net1.module.fc3_3.parameters()) +
                                  list(self.net1.module.fc3_4.parameters()) + list(self.net1.module.fc3_5.parameters()) +
                                  list(self.net2.module.fc3_1.parameters()) + list(self.net2.module.fc3_2.parameters()) + list(self.net2.module.fc3_3.parameters()) +
                                  list(self.net2.module.fc3_4.parameters()) + list(self.net2.module.fc3_5.parameters()) +
                                  list(self.net3.module.fc3_1.parameters()) + list(self.net3.module.fc3_2.parameters()) + list(self.net3.module.fc3_3.parameters()) +
                                  list(self.net3.module.fc3_4.parameters()) + list(self.net3.module.fc3_5.parameters()) +
                                  list(self.net4.module.fc3_1.parameters()) + list(self.net4.module.fc3_2.parameters()) + list(self.net4.module.fc3_3.parameters()) +
                                  list(self.net4.module.fc3_4.parameters()) + list(self.net4.module.fc3_5.parameters()) +
                                  list(self.net5.module.fc3_1.parameters()) + list(self.net5.module.fc3_2.parameters()) + list(self.net5.module.fc3_3.parameters()) +
                                  list(self.net5.module.fc3_4.parameters()) + list(self.net5.module.fc3_5.parameters())))
        base_params = filter(lambda p: id(p) not in ignored_params,
                             list(self.net1.parameters()) + list(self.net2.parameters()) + list(self.net3.parameters()) +
                             list(self.net4.parameters()) + list(self.net5.parameters()))
        self.optimizer = optim.Adam([{'params': base_params},
                                     {'params': self.net1.module.fc3_1.parameters(),
                                      'lr': 0.001},
                                     {'params': self.net1.module.fc3_2.parameters(),
                                      'lr': 0.001},
                                     {'params': self.net1.module.fc3_3.parameters(),
                                      'lr': 0.001},
                                     {'params': self.net1.module.fc3_4.parameters(),
                                      'lr': 0.001},
                                     {'params': self.net1.module.fc3_5.parameters(),
                                      'lr': 0.001},
                                     {'params': self.net2.module.fc3_1.parameters(),
                                      'lr': 0.001},
                                     {'params': self.net2.module.fc3_2.parameters(),
                                      'lr': 0.001},
                                     {'params': self.net2.module.fc3_3.parameters(),
                                      'lr': 0.001},
                                     {'params': self.net2.module.fc3_4.parameters(),
                                      'lr': 0.001},
                                     {'params': self.net2.module.fc3_5.parameters(),
                                      'lr': 0.001},
                                     {'params': self.net3.module.fc3_1.parameters(),
                                      'lr': 0.001},
                                     {'params': self.net3.module.fc3_2.parameters(),
                                      'lr': 0.001},
                                     {'params': self.net3.module.fc3_3.parameters(),
                                      'lr': 0.001},
                                     {'params': self.net3.module.fc3_4.parameters(),
                                      'lr': 0.001},
                                     {'params': self.net3.module.fc3_5.parameters(),
                                      'lr': 0.001},
                                     {'params': self.net4.module.fc3_1.parameters(),
                                      'lr': 0.001},
                                     {'params': self.net4.module.fc3_2.parameters(),
                                      'lr': 0.001},
                                     {'params': self.net4.module.fc3_3.parameters(),
                                      'lr': 0.001},
                                     {'params': self.net4.module.fc3_4.parameters(),
                                      'lr': 0.001},
                                     {'params': self.net4.module.fc3_5.parameters(),
                                      'lr': 0.001},
                                     {'params': self.net5.module.fc3_1.parameters(),
                                      'lr': 0.001},
                                     {'params': self.net5.module.fc3_2.parameters(),
                                      'lr': 0.001},
                                     {'params': self.net5.module.fc3_3.parameters(),
                                      'lr': 0.001},
                                     {'params': self.net5.module.fc3_4.parameters(),
                                      'lr': 0.001},
                                     {'params': self.net5.module.fc3_5.parameters(), 'lr': 0.001}, ], lr=0)
        self.transformer = transforms.Compose([transforms.ToTensor(
        ), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
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

    def evaluate(self, not_valid):
        if self.is_train:
            dataset = self.train_data if not_valid else self.valid_data
        else:
            dataset = self.test_data
        correct = 0
        correct1 = 0
        correct2 = 0
        correct3 = 0
        correct4 = 0
        correct5 = 0
        total = 0
        running_loss = 0.0
        batch_num = int(np.ceil(len(dataset)/self.batch_size))
        print('{} pics, batch_size {}, batch_num {}'.format(
            len(dataset), self.batch_size, batch_num))
        for i in range(batch_num):
            tmp_data = dataset[int(i)*self.batch_size:int(i+1)*self.batch_size]
            # print(len(tmp_data))
            if len(tmp_data) == 0:
                break
            input_x1 = []
            input_x2 = []
            input_x3 = []
            input_x4 = []
            input_x5 = []
            input_y = []
            if self.is_train:
                for img, label in tmp_data:

                    img1 = Imadjust(img)
                    img2 = Histeq(img)
                    img3 = Adaphisteq(img)
                    img4 = Conorm(img)
                    img5 = img
                    if not_valid:
                        img1 = self.distortion(img1)
                        img2 = self.distortion(img2)
                        img3 = self.distortion(img3)
                        img4 = self.distortion(img4)
                        img5 = self.distortion(img5)
                    img1 = cv2.resize(img1, (32, 32))
                    img2 = cv2.resize(img2, (32, 32))
                    img3 = cv2.resize(img3, (32, 32))
                    img4 = cv2.resize(img4, (32, 32))
                    img5 = cv2.resize(img5, (32, 32))
                    input_x1.append(self.transformer(
                        img1).expand(1, 3, 32, 32))
                    input_x2.append(self.transformer(
                        img2).expand(1, 3, 32, 32))
                    input_x3.append(self.transformer(
                        img3).expand(1, 3, 32, 32))
                    input_x4.append(self.transformer(
                        img4).expand(1, 3, 32, 32))
                    input_x5.append(self.transformer(
                        img5).expand(1, 3, 32, 32))
                    input_y.append(label)
            # print(input_x2.shape)
                input_x1 = torch.cat(input_x1, dim=0)
                input_x2 = torch.cat(input_x2, dim=0)
                input_x3 = torch.cat(input_x3, dim=0)
                input_x4 = torch.cat(input_x4, dim=0)
                input_x5 = torch.cat(input_x5, dim=0)
                input_y = np.asarray(input_y)
                if not_valid:
                    self.optimizer.zero_grad()
                    input_y = torch.from_numpy(input_y).cuda()
                    output1 = self.net1(input_x1)
                    output2 = self.net2(input_x2)
                    output3 = self.net3(input_x3)
                    output4 = self.net4(input_x4)
                    output5 = self.net5(input_x5)
                    output = (output1 + output2 + output3 +
                              output4 + output5)/(5 * 5)
                #output = output.long()
                    input_y = input_y.long()
                # print(input_y.shape)
                # print(output.shape)
                    loss = self.criterion(output, input_y)
                    _, predicted = torch.max(output, 1)
                    _, predicted1 = torch.max(output1, 1)
                    _, predicted2 = torch.max(output2, 1)
                    _, predicted3 = torch.max(output3, 1)
                    _, predicted4 = torch.max(output4, 1)
                    _, predicted5 = torch.max(output5, 1)
                    loss.backward()
                    self.optimizer.step()
                else:
                    with torch.no_grad():
                        input_y = torch.from_numpy(input_y).cuda()
                        output1 = self.net1(input_x1)
                        output2 = self.net2(input_x2)
                        output3 = self.net3(input_x3)
                        output4 = self.net4(input_x4)
                        output5 = self.net5(input_x5)
                        output = (output1 + output2 + output3 +
                                  output4 + output5) / (5 * 5)
                        input_y = input_y.long()
                        loss = self.criterion(output, input_y)
                        _, predicted = torch.max(output, 1)
                        _, predicted1 = torch.max(output1, 1)
                        _, predicted2 = torch.max(output2, 1)
                        _, predicted3 = torch.max(output3, 1)
                        _, predicted4 = torch.max(output4, 1)
                        _, predicted5 = torch.max(output5, 1)
                running_loss += loss.item()
                total += input_y.size(0)
                correct += (predicted == input_y).sum().item()
                correct1 += (predicted1 == input_y).sum().item()
                correct2 += (predicted2 == input_y).sum().item()
                correct3 += (predicted3 == input_y).sum().item()
                correct4 += (predicted4 == input_y).sum().item()
                correct5 += (predicted5 == input_y).sum().item()
                print('Current avg loss {}'.format(loss.item()/len(tmp_data)))
            else:
                for img in tmp_data:
                    img1 = Imadjust(img)
                    img2 = Histeq(img)
                    img3 = Adaphisteq(img)
                    img4 = Conorm(img)
                    img5 = img
                    img1 = cv2.resize(img1, (32, 32))
                    img2 = cv2.resize(img2, (32, 32))
                    img3 = cv2.resize(img3, (32, 32))
                    img4 = cv2.resize(img4, (32, 32))
                    img5 = cv2.resize(img5, (32, 32))
                    input_x1.append(self.transformer(
                        img1).expand(1, 3, 32, 32))
                    input_x2.append(self.transformer(
                        img2).expand(1, 3, 32, 32))
                    input_x3.append(self.transformer(
                        img3).expand(1, 3, 32, 32))
                    input_x4.append(self.transformer(
                        img4).expand(1, 3, 32, 32))
                    input_x5.append(self.transformer(
                        img5).expand(1, 3, 32, 32))
                input_x1 = torch.cat(input_x1, dim=0)
                input_x2 = torch.cat(input_x2, dim=0)
                input_x3 = torch.cat(input_x3, dim=0)
                input_x4 = torch.cat(input_x4, dim=0)
                input_x5 = torch.cat(input_x5, dim=0)
                with torch.no_grad():
                    output1 = self.net1(input_x1)
                    output2 = self.net2(input_x2)
                    output3 = self.net3(input_x3)
                    output4 = self.net4(input_x4)
                    output5 = self.net5(input_x5)
                    output = (output1 + output2 + output3 +
                              output4 + output5) / (5 * 5)
                    _, predicted = torch.max(output, 1)
                    self.pred += list(predicted.cpu().numpy())
        correct = [correct, correct1, correct2, correct3, correct4, correct5]
        return correct, total, running_loss

    @staticmethod
    # distort the image in train epoches in order to get better generalize performance
    def distortion(img):
        img_size = np.shape(img)
        resize_ratio = np.random.uniform(0.9, 1.1)  # get a random scale
        # get a random shift amount
        yshift = np.random.uniform(-0.1 * img_size[0], 0.1 * img_size[0])
        xshift = np.random.uniform(-0.1 * img_size[1], 0.1 * img_size[1])
        M = np.float32([[1, 0, round(xshift)], [0, 1, round(yshift)]])
        img = cv2.warpAffine(
            img, M, (img_size[1], img_size[0]))    # random shift
        img = cv2.resize(img, None, fx=resize_ratio,
                         fy=resize_ratio)   # random scale
        img_size = np.shape(img)
        rotate_ratio = np.random.uniform(-5, 5)  # get a random degree
        M = cv2.getRotationMatrix2D(
            (img_size[1] / 2, img_size[0] / 2), rotate_ratio, 1)    # get rotate matrix
        img = cv2.warpAffine(img, M, (img_size[1], img_size[0]))    # rotate
        return img

    def train(self):
        # todo: training process
        print('===========Training started===============')
        best_acc = 0.0
        log_file = open(
            './train_log_{}.txt'.format(time.strftime("%d-%H-%M")), 'w', encoding='utf8')
        for i in range(self.epoch_num):
            correct, total, loss = self.evaluate(not_valid=True)
            print('epoch '+str(int(self.epoch))+' finished, acc {}, loss {}'.format(
                correct[0]/total, loss/len(self.train_data)))
            print('Dataset accs:{} {} {} {} {}'.format(
                correct[1]/total, correct[2]/total, correct[3]/total, correct[4]/total, correct[5]/total))
            log_file.write('{}\tepoch '.format(time.strftime("%d-%H-%M"))+str(int(self.epoch)) +
                           'finished, acc {}, loss {}\n'.format(correct[0]/total, loss/len(self.train_data)))
            log_file.write('Dataset accs:{} {} {} {} {}\n'.format(
                correct[1]/total, correct[2]/total, correct[3]/total, correct[4]/total, correct[5]/total))
            log_file.flush()
            state = {'net1': self.net1.state_dict(), 'net2': self.net2.state_dict(), 'net3': self.net3.state_dict(),
                     'net4': self.net4.state_dict(), 'net5': self.net5.state_dict(),
                     'optimizer': self.optimizer.state_dict(), 'epoch': self.epoch}
            torch.save(state, './task3_MCDNN.pth')
            if (i+1) % self.valid_epochs == 0:
                correct, total, loss = self.evaluate(not_valid=False)
                print('Validation ended, acc {}, loss {}'.format(
                    correct[0] / total, loss/len(self.valid_data)))
                print('Dataset accs:{} {} {} {} {}'.format(correct[1] / total, correct[2] / total, correct[3] / total,
                                                           correct[4] / total, correct[5] / total))
                log_file.write('{}\tValidation ended, acc {}, loss {}\n'.format(
                    time.strftime("%d-%H-%M"), correct[0] / total, loss/len(self.valid_data)))
                log_file.write(
                    'Dataset accs:{} {} {} {} {}\n'.format(correct[1] / total, correct[2] / total, correct[3] / total,
                                                           correct[4] / total, correct[5] / total))
                log_file.flush()
                if (i+1) == self.valid_epochs:
                    best_acc = correct[0]/total
                    print('Best model saved.')
                    log_file.write('Best model saved.\n')
                    log_file.flush()
                    torch.save(state, './task3_MCDNN_best.pth')
                else:
                    acc = correct[0]/total
                    if acc > best_acc:
                        best_acc = acc
                        print('Best model saved.')
                        log_file.write('Best model saved.\n')
                        log_file.flush()
                        torch.save(state, './task3_MCDNN_best.pth')
            self.epoch += 1
        log_file.close()

    def Test(self):
        _, _, _ = self.evaluate(not_valid=False)
        final = {}
        for i, name in enumerate(self.data_name):
            final[name] = self.label_dict_reverse[self.pred[i]]
        json.dump(final, open('./task3_MCDNN.json', 'w', encoding='utf8'))

    def load_best_model(self):
        state = torch.load('./task3_MCDNN_best.pth')
        self.net1.load_state_dict(state['net1'], False)
        self.net2.load_state_dict(state['net2'], False)
        self.net3.load_state_dict(state['net3'], False)
        self.net4.load_state_dict(state['net4'], False)
        self.net5.load_state_dict(state['net5'], False)
        self.optimizer.load_state_dict(state['optimizer'])
        self.epoch = state['epoch']


if __name__ == "__main__":

    test_path = '/home/ass02/Datasets/image_exp/Classification/DataFewShot/Test/'
    classes = ('p1', 'p12', 'p14', 'p17', 'p19',
               'p22', 'p25', 'p27', 'p3', 'p6', 'p9')
    config_dict = {'batch_size': 4,
                   'lr': 0.001,
                   'is_train': False,
                   'epoch_num': 100,
                   'validation_epochs': 5}
    model = Model(data_path=test_path, classes=classes, config=config_dict)
    model.load_best_model()
    model.Test()
