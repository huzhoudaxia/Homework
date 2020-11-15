from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
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
xdata = np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]])
ydata = np.array(xdata, copy=True)
# 将数据做成数据集的模样

x = Variable(torch.Tensor(xdata))
x = x.float()
y = Variable(torch.from_numpy(ydata))
y = y.float()

net = Net(4, 2, 4)

# 定义优化器和损失函数
optim = torch.optim.SGD(Net.parameters(net), lr=0.5, momentum=0.55)
Loss = nn.MSELoss()


# 下面开始训练：
# 一共训练 1000次
x_index = []
for epoch in range(60000):
    loss = None
    # for batch_x, batch_y in dataloader:
    out = net(x)
    loss = Loss(out[1], y)
    optim.zero_grad()
    loss.backward()
    optim.step()
    # 每100次 的时候打印一次日志
    '''
    if (epoch + 1) % 2 == 0:
        print("step: {0} , loss: {1}".format(epoch + 1, loss.item()))
    '''

# 使用训练好的模型进行预测
predict = net(torch.tensor(x, dtype=torch.float))
print('隐层输出', predict[0], '隐层恢复输出值', predict[1])
