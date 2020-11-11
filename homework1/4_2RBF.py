import torch
import numpy as np
import random
import torch.nn as nn
import torch.optim as optim
import math
import matplotlib.pyplot as plt
from torch.autograd import Variable
torch.manual_seed(42)


class RBFN(nn.Module):
    """
    以高斯核作为径向基函数
    """

    def __init__(self, centers, n_out=3):
        """
        :param centers: shape=[center_num,data_dim]
        :param n_out:
        """
        super(RBFN, self).__init__()
        self.n_out = n_out
        self.num_centers = centers.size(0)  # 隐层节点的个数
        self.dim_centure = centers.size(1)
        self.centers = nn.Parameter(centers)
        # self.beta = nn.Parameter(torch.ones(1, self.num_centers), requires_grad=True)
        self.beta = torch.ones(1, self.num_centers) * 10
        # 对线性层的输入节点数目进行了修改
        self.linear = nn.Linear(
            self.num_centers + self.dim_centure, self.n_out, bias=True)
        self.initialize_weights()  # 创建对象时自动执行

    def kernel_fun(self, batches):
        n_input = batches.size(0)  # number of inputs
        A = self.centers.view(self.num_centers, -1).repeat(n_input, 1, 1)
        B = batches.view(n_input, -1).unsqueeze(1).repeat(1,
                                                          self.num_centers, 1)
        C = torch.exp(-self.beta.mul((A - B).pow(2).sum(2, keepdim=False)))
        return C

    def forward(self, batches):
        radial_val = self.kernel_fun(batches)
        class_score = self.linear(torch.cat([batches, radial_val], dim=1))
        return class_score

    def initialize_weights(self, ):
        """
        网络权重初始化
        :return:
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()

    def print_network(self):
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print(self)
        print('Total number of parameters: %d' % num_params)

# centers = torch.rand((5,8))
# rbf_net = RBFN(centers)
# rbf_net.print_network()
# rbf_net.initialize_weights()


if __name__ == "__main__":
    '''
    data = torch.tensor([[0.25, 0.75], [0.75, 0.75], [0.25, 0.5], [0.5, 0.5], [0.75, 0.5],
                         [0.25, 0.25], [0.75, 0.25], [0.5, 0.125], [0.75, 0.125]], dtype=torch.float32)
    label = torch.tensor([[-1, 1, -1], [1, -1, -1], [-1, -1, 1], [-1, -1, 1], [-1, -1, 1],
                          [1, -1, -1], [-1, 1, -1], [-1, 1, -1], [1, -1, -1]], dtype=torch.float32)
                          '''
    # 准备数据
    xdata = np.random.uniform(low=-4., high=4., size=32)
    xdata.sort()
    print(xdata)
    ydata_ori = np.empty(shape=(0))
    noi = np.random.normal(loc=0.0, scale=0.1, size=32)
    for i in range(len(xdata)):
        ydata_ori = np.append(ydata_ori, [1.1 * (1 - xdata[i] + 2 * (xdata[i]**2))
                                          * math.pow(math.e, -0.5 * (xdata[i]**2))], axis=0)
    ydata = ydata_ori + noi
    xdata = np.reshape(xdata, (32, 1))
    ydata = np.reshape(ydata, (32, 1))
    # 将数据做成数据集的模样

    x = Variable(torch.Tensor(xdata))
    data = x.float()
    y = Variable(torch.from_numpy(ydata))
    label = y.float()

    print(data.size())

    centers = data[0:31, :]
    rbf = RBFN(centers, 1)  # 设定输出节点个数为1
    params = rbf.parameters()
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(params, lr=0.2, momentum=0.4)

    for i in range(10000):
        optimizer.zero_grad()

        y = rbf.forward(data)
        loss = loss_fn(y, label)
        loss.backward()
        optimizer.step()
        print(i, "\t", loss.data)

    # 加载使用
    predict = rbf.forward(data)

    print(y.data)
    print(label.data)
    # 使用训练好的模型进行预测

    print(torch.tensor(x, dtype=torch.float))
    print(predict)

    # 绘图展示预测的和真实数据之间的差异
    plt.plot(x, ydata_ori, label="fact")
    plt.plot(x, predict.detach().numpy(), label="predict")
    plt.title("Hermit")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.savefig(fname="result.png", figsize=[10, 10])
    plt.show()
