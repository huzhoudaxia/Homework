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
import cv2
import os
import matplotlib.pyplot as plt  # plt 用于显示图片
import matplotlib.image as mpimg  # mpimg 用于读取图片
import numpy as np


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

    def local_threshold(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        # binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,25,10)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 10)
        # cv2.imshow("binary1", binary)
        return binary

    def extract_imglabel(self, dir):
        '''
        param function:从目录中提取图片数据和标签
        param label数据格式，一维ndarray，
        param pic：3维ndarray，单张图片数据格式为2维

        '''
        files = os.listdir(dir)
        label = []
        pic = np.empty(shape=(0, 40, 20))
        for name in files:
            label.append(name[-5])
            img = cv2.imread(dir + '/' + name)

            tmp = self.local_threshold(img)
            resized = cv2.resize(tmp, (20, 40))  # 裁剪
            '''
            cv2.imshow("binary ", resized)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            '''
            # print(label)

            pic = np.append(pic, [resized], axis=0)
        label_num, label_dict = self.label_dict(label)
        return pic, label_num, label_dict

    def getUniqueItems(self, labels):
        '''
        param function:将一个list中的字符数据去重重新保存输出
        param labels:list,元素为字符数据，1维
        param label：list,元素为labels此list中去重的list，1维
        '''
        label = []
        for item in labels:
            if item not in label:
                label.append(item)
        return label

    def label_dict(self, labels):
        '''
        param function:为每一个标签添加对应的数值，用字典形式存储
        param labels:list,元素为字符数据，1维
        param label_num:1维ndarray数组，储存着labels中对应标签的值
        '''
        label = self.getUniqueItems(labels)
        label_dict = {}
        for i in range(len(label)):
            label_dict[label[i]] = i
        label_num = np.empty(shape=(0))
        for i in range(len(labels)):
            tmp = label_dict[labels[i]]
            label_num = np.append(label_num, [tmp], axis=0)
        return label_num, label_dict


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=2),
            nn.BatchNorm2d(25),
            nn.ReLU(inplace=True)
        )

        self.layer2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(25, 50, kernel_size=2),
            nn.BatchNorm2d(50),
            nn.ReLU(inplace=True)
        )

        self.layer4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc = nn.Sequential(
            nn.Linear(1800, 600),
            nn.ReLU(inplace=True),
            nn.Linear(600, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 41)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


cnn = CNN()
tool = tool()
# 准备数据
x_trai, lable_trai, dict_train = tool.extract_imglabel('./number/train')
x_pred, lable_pred, dict_pred = tool.extract_imglabel('./number/test')
print(len(dict_train))
print(len(dict_pred))
# 转换成tensor
train_data = torch.tensor(x_trai, dtype=torch.float)
# train_data = Variable(train_data)
train_data = train_data.view(len(x_trai), 1, 40, 20)
train_lable = torch.tensor(lable_trai, dtype=torch.int64)
# train_lable = Variable(train_lable)

predict_data = torch.tensor(x_pred, dtype=torch.float)
# predict_data = Variable(predict_data)
predict_data = predict_data.view(len(x_pred), 1, 40, 20)
predict_lable = torch.tensor(lable_pred, dtype=torch.int64)
# predict_lable = Variable(predict_lable)

# 定义一些超参数
batch_size = 8
learning_rate = 0.02
momentum = 0.02
iteration = 5000
# 使用批训练方式


if torch.cuda.is_available():
    cnn = cnn.cuda()

# 定义损失函数和优化器

Loss = nn.CrossEntropyLoss()
optim = torch.optim.Adam(CNN.parameters(
    cnn), lr=learning_rate)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# 使用批训练方式
trainset = TensorDataset(train_data, train_lable)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

predset = TensorDataset(predict_data, predict_lable)
predloader = DataLoader(predset, batch_size=batch_size, shuffle=True)

# 训练模型
epoch = 0
for i in range(iteration):
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
