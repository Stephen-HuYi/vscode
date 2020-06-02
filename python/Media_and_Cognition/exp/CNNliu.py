
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch
import json
import torchvision.transforms as T
from PIL import Image
import os


# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        return x

trans = T.Compose([T.Resize((32, 32)), T.ToTensor(),
                       T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

if __name__ == "__main__":

    net = Net1()
    net_dict = net.state_dict()
    state_dict = torch.load('CNN20.pth')  # 加载预先训练好的.pth文件
    #new_state_dict = OrderedDict()  # 不是必要的【from collections import OrderedDict】
    new_state_dict = {k: v for k, v in state_dict.items() if k in net_dict}  # 删除不需要的键
    net_dict.update(new_state_dict)  # 更新参数
    net.load_state_dict(net_dict)  # 加载参数
    root='/home/ass02/Datasets/image_exp/Classification/DataFewShot/'
    # 读取训练集数据得到每个类别对应的高维向量
    f = open(root+'train.json')
    dict = json.load(f)
    classes=['p17', 'p14',  'p3','p25',  'p9', 'p1',  'p27', 'p19', 'p6',  'p12',  'p22']
    record=[]
    for key in dict:
        filename=(root+'Train/'+dict[key]+'/'+key)
        img = Image.open(filename).convert('RGB')
        input = trans(img)
        input = input.unsqueeze(0)
        output=net(input)
        record.append(output.detach().numpy())
    for i in range(11):
        record[i]=record[i]/np.linalg.norm(record[i])

    #读取测试集数据作为模型输入，将模型输出与以上高维向量对比
    result={}
    f=os.listdir(root+'Test/')
    for name in f:
        filename=(root+'Test/'+name)
        img = Image.open(filename).convert('RGB')
        input = trans(img)
        input = input.unsqueeze(0)
        output = net(input)
        d_min=float('inf')
        for i in range(11):
            temp=output.detach().numpy()
            temp=temp/np.linalg.norm(temp)
            d=np.sqrt(np.sum(np.square(temp-record[i])))
            if d<d_min:
                d_min=d
                index=i
        result[name]=classes[index]
    print(result)
    f = open('test.json', 'w')
    json.dump(result, f)



