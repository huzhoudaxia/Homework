# coding: utf8

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torchvision import datasets, transforms
import ast
from prettytable import PrettyTable
import random
from torch import nn


class tool():
    def __init__(self):
        self.test = None

    def read_txt(self):
        #  文件，
        #  逐行读取
        f = open('./Char7Data.txt')
        data = f.readlines()  # 逐行读取txt并存成list。每行是list的一个元素，数据类型为str
        dot = []
        lable = []
        for i in range(len(data)):  # len(data)为数据行数
            # :: i为偶数表示点阵数据，i为奇数表示标签数据
            # l.append(data[i].split(' ')[j])
            if i % 2 == 0:  # 偶数
                dot.append(ast.literal_eval(data[i]))
            else:
                lable.append(ast.literal_eval(data[i]))
        for i in range(len(lable)):
            if(lable[i] == [1, 0, 0, 0, 0, 0, 0]):
                lable[i] = 0  # 'A'
            elif(lable[i] == [0, 1, 0, 0, 0, 0, 0]):
                lable[i] = 1   # 'B'
            elif(lable[i] == [0, 0, 1, 0, 0, 0, 0]):
                lable[i] = 2  # 'C'
            elif(lable[i] == [0, 0, 0, 0, 0, 0, 1]):
                lable[i] = 3  # 'K'
            elif(lable[i] == [0, 0, 0, 0, 0, 1, 0]):
                lable[i] = 4  # 'J'
            elif(lable[i] == [0, 0, 0, 0, 1, 0, 0]):
                lable[i] = 5  # 'E'
            elif(lable[i] == [0, 0, 0, 1, 0, 0, 0]):
                lable[i] = 6  # 'D'
        return dot, lable

    def add_noise(self, data, percent):
        '''
        param function:在一个一维二值向量(长度63）内随机翻转percent个数据
        # :: data数据结构为ndarray,sam1
        # :: function产生sam的1个噪声样本
        # :: 返回ndarray结构数据，1维结构
        '''
        inde = random.sample(range(63), percent)  # 方法1
        for i in range(len(inde)):
            data[inde[i]] = abs(data[inde[i]] - 1)
        return data

    def add_noise_all(self, data, percent):
        '''
        # :: data数据结构ndarray,样本的点阵数据，一维
        param percent:表示噪声的百分比，这里用噪声点个数表示
        # :: function产生sam的所有噪声样本
        # :: 返回ndarray，二维结构，元素为ndarray
        '''
        noise = np.empty(shape=(0, 25))
        for i in range(len(data)):
            tmp = self.add_noise(data, i, percent)
            noise = np.append(noise, [tmp], axis=0)
        return noise

    def make_samples(self, x, percent):
        # :: x为正确的一组输入样本，21*63，数据结构ndarray，[x1,x2,...]
        # :: function为产生这组样本的噪声样本
        # :: 返回值用ndarray，shape:
        noise = np.empty(shape=(0, 63))  # 最终结构应该是一个三维矩阵，里面元素是每个样本的多个噪声样本
        for i in range(len(x)):
            data = x[i]  # 单个样本矩阵
            noise_tmp = self.add_noise(data, percent)  # 一维矩阵ndarray
            noise = np.append(noise, [noise_tmp], axis=0)
        return noise


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=3),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )

        self.layer2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc = nn.Sequential(
            nn.Linear(2 * 2 * 3, 14),
            nn.ReLU(inplace=True),
            nn.Linear(14, 11),
            nn.Sigmoid(),
            nn.Linear(11, 7),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        # x.size()为batchsize，其它维度reshape成一个向量，即batch个样本，每个样本的长度为1维度，再与fc的输入层全连接
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


cnn = CNN()
tool = tool()
# 准备数据
x, lable = tool.read_txt()
# 转换成tensor
train_data = torch.tensor(x[0:14], dtype=torch.float)
#train_data = Variable(train_data)
train_data = train_data.view(14, 1, 9, 7)
train_lable = torch.tensor(lable[0:14], dtype=torch.int64)
#train_lable = Variable(train_lable)

predict_data = torch.tensor(x[14:21], dtype=torch.float)
#predict_data = Variable(predict_data)
predict_data = predict_data.view(7, 1, 9, 7)
predict_lable = torch.tensor(lable[14:21], dtype=torch.int64)
#predict_lable = Variable(predict_lable)

# 定义一些超参数
batch_size = 2
learning_rate = 0.02
momentum = 0.01
num_epoches = 20
iteration = 2000
# 使用批训练方式


if torch.cuda.is_available():
    cnn = cnn.cuda()

# 定义损失函数和优化器
optim = torch.optim.SGD(CNN.parameters(
    cnn), lr=learning_rate, momentum=momentum)
Loss = nn.CrossEntropyLoss()
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# 训练模型
epoch = 0
for i in range(iteration):
    # 使用批训练方式
    trainset = TensorDataset(train_data, train_lable)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    predset = TensorDataset(predict_data, predict_lable)
    predloader = DataLoader(predset, batch_size=2, shuffle=True)

    for data in trainloader:
        img, label = data
        if torch.cuda.is_available():
            img = img.cuda()
            label = label.cuda()
        else:
            img = Variable(img)
            label = Variable(label)
        out = cnn(img)
        loss = Loss(out, label)
        print_loss = loss.data.item()

        optim.zero_grad()
        loss.backward()
        optim.step()

        epoch += 1
        if epoch % 50 == 0:
            print('epoch: {}, loss: {:.4}'.format(epoch, loss.data.item()))

# 模型评估
cnn.eval()
eval_loss = 0
eval_acc = 0
for data in predloader:
    img, label = data
    img = Variable(img)
    if torch.cuda.is_available():
        img = img.cuda()
        label = label.cuda()

    out = cnn(img)
    loss = Loss(out, label)
    eval_loss += loss.data.item() * label.size(0)
    _, pred = torch.max(out, 1)
    num_correct = (pred == label).sum()
    eval_acc += num_correct.item()
print('Test Loss: {:.6f}, Acc: {:.6f}'.format(
    eval_loss / (len(predict_data)),
    eval_acc / (len(predict_data))
))
