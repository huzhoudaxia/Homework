from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import pandas as pd
import torch
import math

# 神经网络主要结构，这里就是一个简单的线性结构


class Net(nn.Module):
    def __init__(self, in_num, hidden_num, out_num):
        super(Net, self).__init__()
        # 构建一个两层网络，并且随机话初始权重，给以输入数据后返回矩阵乘积数据。
        self.input_layer = torch.nn.Linear(in_num, hidden_num)
        self.sigmoid = torch.nn.Sigmoid()
        self.output_layer = torch.nn.Linear(hidden_num, out_num)
        self.relu = torch.nn.Sigmoid()

    def forward(self, input_x):
        # 对输入数据与权重矩阵的积进行激活，返回激活输出
        h_1 = self.sigmoid(self.input_layer(input_x))
        h_2 = self.output_layer(h_1)
        return h_1, h_2


# 准备数据
xdata = np.empty(shape=(0, 63))
a = np.array([[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0,
               0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]])
xdata = np.append(xdata, a, axis=0)
b = np.array([[1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1,
               0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
xdata = np.append(xdata, b, axis=0)
c = np.array([[0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1,
               0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
xdata = np.append(xdata, c, axis=0)
d = np.array([[1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1,
               0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
xdata = np.append(xdata, d, axis=0)
e = np.array([[1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1,
               0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]])
xdata = np.append(xdata, e, axis=0)
f = np.array([[1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1,
               0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
xdata = np.append(xdata, f, axis=0)
g = np.array([[0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0,
               0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
xdata = np.append(xdata, g, axis=0)
h = np.array([[1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1,
               0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]])
xdata = np.append(xdata, h, axis=0)
i = np.array([[0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
               0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]])
xdata = np.append(xdata, i, axis=0)
j = np.array([[0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
               0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
xdata = np.append(xdata, j, axis=0)
label = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
ydata = np.array(xdata, copy=True)
# 将数据做成数据集的模样

x = Variable(torch.Tensor(xdata))
x = x.float()
y = Variable(torch.from_numpy(ydata))
y = y.float()

epoch_list = []
hidd_list = []
hidd = 10
test = 100
epoch = 0

flage = True   # 重新定义网络的标志
if(flage):
    net = Net(63, hidd, 63)
    # 定义优化器和损失函数
    optim = torch.optim.SGD(Net.parameters(net), lr=0.1, momentum=0.4)
    Loss = nn.MSELoss()
    flage = False
# 下面开始训练：
while(test):
    if(flage):
        net = Net(63, hidd, 63)
        # 定义优化器和损失函数
        optim = torch.optim.SGD(Net.parameters(net), lr=0.1, momentum=0.4)
        Loss = nn.MSELoss()
        flage = False
    x_index = []
    #iteration = 10000
    # for epoch in range(iteration):
    loss = None
    # for batch_x, batch_y in dataloader:
    out = net(x)
    loss = Loss(out[1], y)
    optim.zero_grad()
    loss.backward()
    optim.step()
    # 每100次 的时候打印一次日志

    if (epoch + 1) % 2 == 0:
        print("step: {0} , loss: {1}".format(epoch + 1, loss.item()))

    # 输出误差在0.1以下，终止训练，打印出epoch

    if(loss.item() <= 0.1):
        epoch_list.append(epoch)
        hidd_list.append(hidd)
        epoch = 0
        hidd += 1
        test -= 1
        flage = True
    else:
        epoch += 1
    print("test", test, "epoch", epoch)
    # 画出隐层节点个数与训练迭代次数的关系


plt.scatter(hidd_list, epoch_list)
plt.xlabel("hidden layer num")
plt.ylabel("iteration num")
plt.show()


net = Net(63, 15, 63)
# 定义优化器和损失函数
optim = torch.optim.SGD(Net.parameters(net), lr=0.1, momentum=0.4)
Loss = nn.MSELoss()
for epoch in range(10000):
    loss = None
    # for batch_x, batch_y in dataloader:
    out = net(x)
    loss = Loss(out[1], y)
    optim.zero_grad()
    loss.backward()
    optim.step()
# 使用训练好的模型进行预测
predict = net(torch.tensor(x, dtype=torch.float))
print('隐层输出', predict[0])
# 恢复图像
hidd_out = []
for i in range(len(predict[1])):
    print(predict[1][i])
    test = abs(predict[1][i] * 255 - 255.)
    # print(test)
    test = np.reshape(test.detach().numpy(), [9, 7])
    plt.imshow(test, cmap='gray')
    plt.title(label[i])
    plt.savefig(fname=label[i] + ".png", figsize=[9, 7])
    plt.show()
