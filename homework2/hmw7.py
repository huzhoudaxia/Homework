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
    # xdata = np.linspace(0., 1.5, 500)
    num1 = 30
    num2 = 80
    num3 = 80
    num4 = 30
    num = num1 + num2 + num3 + num4
    iteration = 180
    data1 = np.random.uniform(low=-4, high=-2., size=num1)
    data2 = np.random.uniform(low=-2, high=0., size=num2)
    data3 = np.random.uniform(low=0., high=2., size=num3)
    data4 = np.random.uniform(low=2., high=4., size=num4)
    xdata = np.empty(shape=(0))
    xdata = np.append(xdata, data1, axis=0)
    xdata = np.append(xdata, data2, axis=0)
    xdata = np.append(xdata, data3, axis=0)
    xdata = np.append(xdata, data4, axis=0)
    xdata.sort()
    ydata = np.empty(shape=(0))
    for i in range(len(xdata)):
        ydata = np.append(ydata, [1.1 * (1 - xdata[i] + 2 * (xdata[i]**2))
                                  * math.pow(math.e, -0.5 * (xdata[i]**2))], axis=0)

    xdata = np.reshape(xdata, (num, 1))
    ydata = np.reshape(ydata, (num, 1))
    w = np.array([np.random.uniform(-1.5, 1.5, 1) for i in range(20)])
    w_out = np.array([np.random.uniform(-2.5, 2.5, 20) for i in range(1)])
    # print(w_out)
    # x_norm = tool.Normalize_new(xdata)
    # w_norm = tool.Normalize_new(w)
    tool.dot2 = (iteration, 0.0)
    tool.dot1 = (0., 0.4)
    tool.iteration = iteration
    for ite in range(tool.iteration):  # 第一阶段
        for i in range(len(xdata)):
            dis = tool.Euclidean(xdata[i], w)
            data, inde = tool.find_winner(dis)
            # w[inde] = tool.WTA(xdata[i], w[inde], 0, ite)

            if(inde > 0 and inde < len(w) - 1):
                w[inde] = tool.WTA(xdata[i], w[inde], 0, ite)
                w[inde - 1] = tool.WTA(xdata[i], w[inde - 1], 1, ite)
                w[inde + 1] = tool.WTA(xdata[i], w[inde + 1], 1, ite)
            elif(inde == 0):
                w[inde] = tool.WTA(xdata[i], w[inde], 1, ite)
            elif(inde == len(w) - 1):
                w[inde] = tool.WTA(xdata[i], w[inde], 1, ite)
            # w_norm = tool.Normalize_new(w)
    # print(w)
    tool.dot2 = (iteration, 0.0)
    tool.dot1 = (0., 0.4)
    tool.iteration = iteration
    # 调整第二阶段的学习率
    # np.random.shuffle(xdata)
    for ite in range(tool.iteration):    # 第二阶段调整外星向量
        for i in range(len(xdata)):
            dis = tool.Euclidean(xdata[i], w)  # 确定竞争层竞争获胜神经元
            data, inde = tool.find_winner(dis)
            hout = np.array([0 for i in range(len(w))])
            hout[inde] = 1.
            # print(hout)
            yout = tool.neuron(hout, w_out)
            # print(yout, w_out[0][inde])
            #print(ydata[i], hout, yout, w_out[inde])
            for j in range(len(w_out)):  # 确定有多少个输出神经元，调整隐层获胜神经元与每个输出神经元的权重
                w_out[j][inde] = tool.out_WTA(
                    ydata[i][j], yout[j], w_out[j][inde], 0, ite)

    # 验证网络
    y_out = np.empty(shape=(0), dtype=float)
    for i in range(len(xdata)):
        dis = tool.Euclidean(xdata[i], w)  # 确定竞争层竞争获胜神经元
        data, inde = tool.find_winner(dis)
        hout = np.array([0 for i in range(len(w))])
        hout[inde] = 1.
        tmp = tool.neuron(hout, w_out)
        y_out = np.append(y_out, tmp, axis=0)  # 保存输出神经元组的第一个神经元的输出数据
        '''
        hout = tool.neuron_act(xdata[i], w)
        tmp = tool.neuron(hout, w_out)
        y_out = np.append(y_out, tmp, axis=0)  # 保存输出神经元组的第一个神经元的输出数据
        '''
    x = np.reshape(xdata, (num))
    y = np.reshape(ydata, (num))

    plt.plot(x, y, label="fact")
    plt.plot(x, y_out, label="predict")
    plt.title("Hermit function")
    plt.xlabel("x")
    plt.ylabel("Her")
    plt.legend()
    # plt.savefig(fname="result.png", figsize=[10, 10])
    plt.show()


##########################################################################
