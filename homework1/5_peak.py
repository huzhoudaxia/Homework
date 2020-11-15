from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.autograd import Variable
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import torch
import math

# 神经网络主要结构，这里就是一个简单的线性结构


class Net(nn.Module):
    '''
    def __init__(self):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=1, out_features=15), nn.Sigmoid(),
            nn.Linear(10, 100), nn.ReLU(),
            nn.Linear(100, 10), nn.ReLU(),
            nn.Linear(10, 1)
        )

    def forward(self, input: torch.FloatTensor):
        return self.net(input)
    '''

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
        return h_2


# 准备数据
x = np.random.uniform(low=-4., high=4., size=100)
y = np.random.uniform(low=-4., high=4., size=100)
x.sort()
y.sort()
# 生成mesh网格点坐标矩阵
x, y = np.meshgrid(x, y)

z = 3 * (1 - x) ** 2. * np.exp(-(x ** 2) - (y + 1) ** 2) - 10 * (x /
                                                                 5 - x ** 3 - y**5) * np.exp(-x**2 - y**2) - 1 / 3 * np.exp(-(x + 1)**2 - y**2)
xdata = x.reshape((10000))
ydata = y.reshape((10000))
zdata = z.reshape((10000, 1))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# 或者ax = axes3D(fig)
# Plot a basic wireframe.
ax.plot_surface(x, y, z)
# ax.view_init(elev=300, azim=300)
# plt.show()
plt.title("fact peaks")

# 将数据做成数据集的模样
x_set = Variable(torch.Tensor(xdata))
x_set = x_set.float()
y_set = Variable(torch.from_numpy(ydata))
y_set = y_set.float()

input_x = np.empty(shape=(0, 2))
for i in range(len(x_set)):
    input_x = np.append(input_x, [[x_set[i], y_set[i]]], axis=0)

input_x = Variable(torch.from_numpy(input_x))
input_x = input_x.float()

zdata = Variable(torch.from_numpy(zdata))
zdata = zdata.float()
'''
# 使用批训练方式
dataset = TensorDataset(torch.tensor(X, dtype=torch.float),
                        torch.tensor(Y, dtype=torch.float))
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
'''
net = Net(2, 30, 1)

# 定义优化器和损失函数
optim = torch.optim.SGD(Net.parameters(net), lr=0.2, momentum=0.4)
Loss = nn.L1Loss()


# 下面开始训练：
# 一共训练 1000次
x_index = []
for epoch in range(10000):
    loss = None
    # for batch_x, batch_y in dataloader:
    out = net(input_x)
    loss = Loss(out, zdata)
    optim.zero_grad()
    loss.backward()
    optim.step()
    # 每100次 的时候打印一次日志
    '''
    if (epoch + 1) % 2 == 0:
        print("step: {0} , loss: {1}".format(epoch + 1, loss.item()))
    '''

# 使用训练好的模型进行预测
predict = net(torch.tensor(input_x, dtype=torch.float))
# 绘图展示预测的和真实数据之间的差异
z = predict.detach().numpy()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# 或者ax = axes3D(fig)
# Plot a basic wireframe.
x = x.reshape((100, 100))
y = y.reshape((100, 100))
z = z.reshape((100, 100))
ax.plot_surface(x, y, z)
plt.title("predict peaks")
# ax.view_init(elev=300, azim=300)
plt.show()
