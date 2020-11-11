import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt


class network(torch.nn.Module):
    def __init__(self, in_num, hidden_num, out_num):
        super(network, self).__init__()
        # 构建一个两层网络，并且随机话初始权重，给以输入数据后返回矩阵乘积数据。
        self.input_layer = torch.nn.Linear(in_num, hidden_num)
        self.sigmoid = torch.nn.Sigmoid()
        self.output_layer = torch.nn.Linear(hidden_num, out_num)
        self.tanh = torch.nn.Tanh()

    def forward(self, input_x):
        # 对输入数据与权重矩阵的积进行激活，返回激活输出
        h_1 = self.sigmoid(self.input_layer(input_x))
        h_2 = self.tanh(self.output_layer(h_1))
        return h_2


x = np.zeros([4, 2])
y = np.zeros([4, 1])
for i in range(0, 4):
    x1 = (int)(i / 2)
    x2 = i % 2
    x[i] = np.array([x1, x2])
    y[i] = [x1 ^ x2]
    print(y)
input_x = Variable(torch.Tensor(x))
print(input_x.dtype)
input_x = input_x.float()
print(input_x.dtype)
y = Variable(torch.from_numpy(y))
print(type(y))
print(y.dtype)
y = y.float()
print(y.dtype)
net = network(2, 2, 1)
print(list(net.parameters()))

loss_function = torch.nn.MSELoss()      #
# 构建好神经网络后，网络的参数都保存在parameters()函数当中，权重参数
optimizer = torch.optim.SGD(net.parameters(), lr=0.9, momentum=0.9)
x_index = []
y_dot = [[], [], [], []]
for i in range(10000):
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

plt.subplot(221)
plt.title('0')
plt.plot(x_index, y_dot[0])
plt.subplot(222)
plt.title('1')
plt.plot(x_index, y_dot[1])
plt.subplot(223)
plt.title('1')
plt.plot(x_index, y_dot[2])
plt.subplot(224)
plt.title('0')
plt.plot(x_index, y_dot[3])
plt.show()

out = net(input_x)   # net训练好了，现在进行预测得到预测的结果
print(out)
print(y)

# 双极性输入输出
x = np.array([[-1., -1.], [1., -1.], [-1., 1.], [1., 1.]])
y = np.array([-1., 1., 1., -1.])

input_x = Variable(torch.Tensor(x))
print(input_x.dtype)
input_x = input_x.float()
print(input_x.dtype)
y = Variable(torch.from_numpy(y))
print(type(y))
print(y.dtype)
y = y.float()
print(y.dtype)
net = network(2, 2, 1)
print(list(net.parameters()))

loss_function = torch.nn.MSELoss()      #
# 构建好神经网络后，网络的参数都保存在parameters()函数当中，权重参数
optimizer = torch.optim.SGD(net.parameters(), lr=5e-1, momentum=0.9)
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

plt.subplot(221)
plt.title('-1')
plt.plot(x_index, y_dot[0])
plt.subplot(222)
plt.title('1')
plt.plot(x_index, y_dot[1])
plt.subplot(223)
plt.title('1')
plt.plot(x_index, y_dot[2])
plt.subplot(224)
plt.title('-1')
plt.plot(x_index, y_dot[3])
plt.show()

out = net(input_x)   # net训练好了，现在进行预测得到预测的结果
print(out)
print(y)
