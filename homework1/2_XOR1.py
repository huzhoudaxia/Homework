import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt


class tool_bp():
    def __init__(self):
        self.test = None
        self.lean = None

    def forward(self, d_in, w, d_out):
        ''' 前向传播算法
        param d_in:输入数据，数据结构ndarray，1维
        param d_out:输出数据，数据结构，实数
        param w:权重矩阵，数据结构ndarray,1维

        '''
        net = np.inner(d_in, w)
        d_out = self.active_func(net)
        return d_out

    def backward(self, d_in, delt, w):
        '''反向传播算法
        param d_in:权重的前向输入数据（也即上一神经元的输出），数据结构:实数
        param delt:反向传来的误差信号 数据结构：1维ndarray
        param w: 调整神经元的输出权重的权重矩阵(不是调整权重)，数据结构：1维ndarray
        '''
        net_delt = self.rev_delt(delt)
        delt_w = self.lean * net * d_in
        return delt_w

    def rev_delt(self, delt, d_in, w):
        '''


        '''
        # net_delt = np.inner(delt, w)
        net_delt = d_in * (1 - d_in) * delt * w
        return net_delt


class network(torch.nn.Module):
    def __init__(self, in_num, hidden_num, hidden2_num, out_num):
        super(network, self).__init__()
        # 构建一个两层网络，并且随机话初始权重，给以输入数据后返回矩阵乘积数据。
        self.input_layer = torch.nn.Linear(in_num, hidden_num)
        self.sigmoid = torch.nn.Sigmoid()
        self.hidden2_layer = torch.nn.Linear(hidden_num, hidden2_num)
        self.output_layer = torch.nn.Linear(hidden2_num, out_num)
        self.tanh = torch.nn.Tanh()

    def forward(self, input_x):
        # 对输入数据与权重矩阵的积进行激活，返回激活输出
        h_1 = self.sigmoid(self.input_layer(input_x))
        h_2 = self.tanh(self.output_layer(h_1))
        return h_2


x = np.array([[0.25, 0.75], [0.75, 0.75], [0.25, 0.5], [0.5, 0.5], [0.75, 0.5],
              [0.25, 0.25], [0.75, 0.25], [0.5, 0.125], [0.75, 0.125]])
y = np.array([[-1, 1, -1], [1, -1, -1], [-1, -1, 1], [-1, -1, 1], [-1, -1, 1],
              [1, -1, -1], [-1, 1, -1], [-1, 1, -1], [1, -1, -1]], dtype=float)

input_x = Variable(torch.Tensor(x))
print(input_x.dtype)
input_x = input_x.float()
print(input_x.dtype)
y = Variable(torch.from_numpy(y))
print(type(y))
print(y.dtype)
y = y.float()
print(y.dtype)
net = network(2, 20, 20, 3)
print(list(net.parameters()))

loss_function = torch.nn.MSELoss()      #
# 构建好神经网络后，网络的参数都保存在parameters()函数当中，权重参数
optimizer = torch.optim.SGD(net.parameters(), lr=0.9, momentum=0.9)
x_index = []
y_dot = [[], [], [], []]
for i in range(5000):
    x_index.append(i)
    out = net(input_x)
    loss = loss_function(out, y)
    #print ("loss is %f" % loss.data.numpy())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    y_dot[0].append(out[0][0])
    y_dot[1].append(out[1][0])
    y_dot[2].append(out[2][0])
    y_dot[3].append(out[3][0])

out = net(input_x)   # net训练好了，现在进行预测得到预测的结果
print(out.data, y.data)
