from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import torch


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
        self.relu = torch.nn.ReLU()

    def forward(self, input_x):
        # 对输入数据与权重矩阵的积进行激活，返回激活输出
        h_1 = self.sigmoid(self.input_layer(input_x))
        h_2 = self.output_layer(h_1)
        return h_2


# 准备数据
x = np.linspace(0., 1., 500)
y = np.sin(6 * np.pi * x)
# 将数据做成数据集的模样
X = np.expand_dims(x, axis=1)

Y = y.reshape(500, -1)
x = Variable(torch.Tensor(X))
x = x.float()
y = Variable(torch.from_numpy(Y))
y = y.float()
# 使用批训练方式
dataset = TensorDataset(torch.tensor(X, dtype=torch.float),
                        torch.tensor(Y, dtype=torch.float))
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

net = Net(1, 30, 1)

# 定义优化器和损失函数
optim = torch.optim.SGD(Net.parameters(net), lr=0.2, momentum=0.4)
Loss = nn.L1Loss()


# 下面开始训练：
# 一共训练 1000次
x_index = []
for epoch in range(600000):
    loss = None
    # for batch_x, batch_y in dataloader:
    out = net(x)
    loss = Loss(out, y)
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
print(torch.tensor(x, dtype=torch.float))
print(predict)
# 绘图展示预测的和真实数据之间的差异
plt.plot(x, y, label="fact")
plt.plot(x, predict.detach().numpy(), label="predict")
plt.title("sin function")
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.legend()
plt.savefig(fname="result.png", figsize=[10, 10])
plt.show()
