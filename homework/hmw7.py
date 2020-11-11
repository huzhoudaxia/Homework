# coding=UTF-8
import numpy as np
import math
import random  # np的np.random.randint(0, 25, 5)可能生成相等的随机数
import matplotlib.pyplot as plt
import ast
from prettytable import PrettyTable
from tool import tool


if __name__ == '__main__':
    # 题目7
    tool = tool()
    xdata = np.linspace(-4., 4., 100)
    ydata = np.empty(shape=(0))
    for i in range(len(xdata)):
        ydata = np.append(ydata, [1.1 * (1 - xdata[i] + 2 * (xdata[i]**2))
                                  * math.pow(math.e, -0.5 * (xdata[i]**2))], axis=0)

    xdata = np.reshape(xdata, (100, 1))
    ydata = np.reshape(ydata, (100, 1))
    w = np.array([np.random.rand(1) for i in range(20)])
    w_out = np.array([np.random.rand(20) for i in range(1)])
    # print(w_out)
    # x_norm = tool.Normalize_new(xdata)
    # w_norm = tool.Normalize_new(w)
    tool.dot2 = (1000, 0.0)
    tool.dot1 = (0., 0.5)
    tool.iteration = 1000
    for ite in range(tool.iteration):  # 第一阶段
        for i in range(len(xdata)):
            dis = tool.Euclidean(xdata[i], w)
            data, inde = tool.find_winner(dis)
            w[inde] = tool.WTA(xdata[i], w[inde], 0, ite)
            # w_norm = tool.Normalize_new(w)
    # print(w)
    tool.dot2 = (10000, 0.0)
    tool.dot1 = (0., 0.5)
    tool.iteration = 10000
    # 调整第二阶段的学习率
    # np.random.shuffle(xdata)
    for ite in range(tool.iteration):    # 第二阶段调整外星向量
        for i in range(len(xdata)):
            dis = tool.Euclidean(xdata[i], w)  # 确定竞争层竞争获胜神经元
            data, inde = tool.find_winner(dis)
            hout = np.array([0 for i in range(len(w))])
            hout[inde] = 1.

            yout = tool.neuron(hout, w_out)
            #print(ydata[i], hout, yout, w_out[inde])
            for j in range(len(w_out)):  # 确定有多少个输出神经元，调整隐层获胜神经元与每个输出神经元的权重
                w_out[j][inde] = tool.out_WTA(
                    ydata[i][j], yout[j], w_out[j][inde], 0, ite)

    # 验证网络
    y_out = np.empty(shape=(0), dtype=float)
    for i in range(len(xdata)):
        hout = tool.neuron_act(xdata[i], w)
        tmp = tool.neuron(hout, w_out)
        y_out = np.append(y_out, tmp, axis=0)  # 保存输出神经元组的第一个神经元的输出数据

    print(y_out)
    x = np.reshape(xdata, (100))
    y = np.reshape(ydata, (100))

    plt.plot(x, y, label="fact")
    plt.plot(x, y_out, label="predict")
    plt.title("Hermit function")
    plt.xlabel("x")
    plt.ylabel("Her")
    plt.legend()
    # plt.savefig(fname="result.png", figsize=[10, 10])
    plt.show()
