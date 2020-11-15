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


# 定义一些超参数
batch_size = 64
learning_rate = 0.02
num_epoches = 20
# 数据预处理。transforms.ToTensor()将图片转换成PyTorch中处理的对象Tensor,并且进行标准化（数据在0~1之间）
# transforms.Normalize()做归一化。它进行了减均值，再除以标准差。两个参数分别是均值和标准差
# transforms.Compose()函数则是将各种预处理的操作组合到了一起
data_tf = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0.5], [0.5])])

# 数据集的下载器
train_dataset = datasets.MNIST(
    root='./data', train=True, transform=data_tf, download=True)

test_dataset = datasets.MNIST(root='./data', train=False, transform=data_tf)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
for data in train_loader:
    img, label = data
    # img = img.view(img.size(0), -1)
    # img = Variable(img)
    break

batchsize, channels, height, width = img.data.size()
img.view(batchsize, channels, height, width)
print(img.data.size())
