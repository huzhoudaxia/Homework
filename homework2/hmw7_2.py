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
##########################################################################
# 构造双向CPN
    num1 = 30
    num2 = 80
    num3 = 80
    num4 = 30
    num = num1 + num2 + num3 + num4
    iteration = 2
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
    xydata = np.empty(shape=(0, 2))
    for i in range(len(xdata)):
        xydata = np.append(xydata, [[xdata[i], 1.1 * (1 - xdata[i] + 2 * (xdata[i]**2))
                                     * math.pow(math.e, -0.5 * (xdata[i]**2))]], axis=0)

    print('xydata', xydata)
    w = np.array([np.random.uniform(0, 1.5, 2) for i in range(20)])
    w_out = np.array([np.random.uniform(0, 2.5, 20) for i in range(2)])
    # w = np.array([np.random.rand(2) for i in range(20)])
    # w_out = np.array([np.random.rand(20) for i in range(2)])
    # print(w_out)
    # x_norm = tool.Normalize_new(xdata)
    # w_norm = tool.Normalize_new(w)
    tool.dot2 = (iteration, 0.01)
    tool.dot1 = (0., 0.3)
    tool.iteration = iteration
    for ite in range(tool.iteration):  # 第一阶段
        for i in range(len(xydata)):
            dis = tool.Euclidean(xydata[i], w)
            data, inde = tool.find_winner(dis)
            # w[inde] = tool.WTA(xydata[i], w[inde], 0, ite)

            if(inde > 0 and inde < len(w) - 1):
                w[inde] = tool.WTA(xdata[i], w[inde], 0, ite)
                w[inde - 1] = tool.WTA(xdata[i], w[inde - 1], 1, ite)
                w[inde + 1] = tool.WTA(xdata[i], w[inde + 1], 1, ite)
            elif(inde == 0):
                w[inde] = tool.WTA(xdata[i], w[inde], 1, ite)

            elif(inde == len(w) - 1):
                w[inde] = tool.WTA(xdata[i], w[inde], 1, ite)

    # print(w)
    tool.dot2 = (iteration, 0.01)
    tool.dot1 = (0., 0.3)
    tool.iteration = iteration
    # 调整第二阶段的学习率
    # np.random.shuffle(xdata)
    for ite in range(tool.iteration):    # 第二阶段调整外星向量
        for i in range(len(xydata)):
            dis = tool.Euclidean(xydata[i], w)  # 确定竞争层竞争获胜神经元
            data, inde = tool.find_winner(dis)
            hout = np.array([0 for i in range(len(w))])
            hout[inde] = 1.

            yout = tool.neuron(hout, w_out)
            #print(ydata[i], hout, yout, w_out[inde])
            for j in range(len(w_out)):  # 确定有多少个输出神经元，调整隐层获胜神经元与每个输出神经元的权重
                w_out[j][inde] = tool.out_WTA(
                    xydata[i][j], yout[j], w_out[j][inde], 0, ite)

    # 验证网络
    y_out = np.empty(shape=(0, 2), dtype=float)
    for i in range(len(xydata)):
        #print('xydata', xydata[i], w)
        tmp = np.array(xydata[i], copy=True)
        tmp[1] = 0.
        hout = tool.neuron_act(tmp, w)
        tmp2 = tool.neuron(hout, w_out)
        y_out = np.append(y_out, [tmp2], axis=0)  # 保存输出神经元组的第一个神经元的输出数据

    print(y_out)
    xydata = xydata.T
    y_out = y_out.T
    x = xydata[0]
    y = xydata[1]
    print('+++', x, y)
    plt.plot(x, y, label="fact")
    plt.plot(x, y_out[1], label="predict")
    plt.title("Hermit function")
    plt.xlabel("x")
    plt.ylabel("Her")
    plt.legend()
    # plt.savefig(fname="result.png", figsize=[10, 10])
    plt.show()
