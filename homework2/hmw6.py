# coding=UTF-8
import numpy as np
import math
import random  # np的np.random.randint(0, 25, 5)可能生成相等的随机数
import matplotlib.pyplot as plt
import ast
from prettytable import PrettyTable
from tool import tool


if __name__ == '__main__':
    # 题目6
    tool = tool()
    x = np.array([[246, 53], [408, 79], [909, 89], [115, 264], [396, 335], [
                 185, 456], [699, 252], [963, 317], [922, 389], [649, 515]])
    w = np.array([np.random.randint(100, 200, 2) for i in range(10)])

    w = w.astype(np.float)
    x_norm = tool.Normalize_new(x)
    w_norm = tool.Normalize_new(w)

    # 粗调
    tool.dot2 = (1000., 0.4)
    tool.dot1 = (0., 0.5)
    tool.iteration = 1000
    for ite in range(tool.iteration):
        for i in range(len(x)):
            dis = tool.Euclidean(x[i], w)
            data, inde = tool.find_winner(dis)
            if(inde > 0 and inde < len(w) - 1):
                w[inde] = tool.WTA(x[i], w[inde], 0, ite)
                w[inde - 1] = tool.WTA(x[i], w[inde - 1], 1, ite)
                w[inde + 1] = tool.WTA(x[i], w[inde + 1], 1, ite)
            elif(inde == 0):
                w[inde] = tool.WTA(x[i], w[inde], 1, ite)
                w[len(w) - 1] = tool.WTA(x[i], w[len(w) - 1], 1, ite)
            elif(inde == len(w) - 1):
                w[inde] = tool.WTA(x[i], w[inde], 1, ite)
                w[0] = tool.WTA(x[i], w[0], 1, ite)
            # w_norm = tool.Normalize_new(w)

    print(w.dtype)
    print(w)

    # N=0，粗调节
    tool.dot2 = (500., 0.4)
    tool.dot1 = (0., 0.5)
    tool.iteration = 500
    for ite in range(tool.iteration):
        for i in range(len(x)):
            dis = tool.Euclidean(x[i], w)
            data, inde = tool.find_winner(dis)
            w[inde] = tool.WTA(x[i], w[inde], 0, ite)
            # w_norm = tool.Normalize_new(w)

    # N=0，细调节
    tool.dot2 = (900., 0.05)
    tool.dot1 = (0., 0.4)
    tool.iteration = 1000
    for ite in range(tool.iteration):
        for i in range(len(x)):
            dis = tool.Euclidean(x[i], w)
            data, inde = tool.find_winner(dis)
            w[inde] = tool.WTA(x[i], w[inde], 0, ite)
            # w_norm = tool.Normalize_new(w)
        '''
    winner = []
    for i in range(len(x)):
        out = tool.neuron(x[i], w)
        winner.append(list(out).index(max(out)))
    print(winner)
        '''

# 识别
    # 归一化
    input_norm = tool.Normalize_new(x)
    winner = np.empty(shape=(0), dtype=int)

    for i in range(len(x)):
        dis = tool.Euclidean(x[i], w)
        data, inde = tool.find_winner(dis)
        winner = np.append(winner, [inde], axis=0)
        print(winner)
    # 将 坐标与靠近向量间的示意图画出
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签

    winner = np.unique(winner)
    winner2 = np.sort(winner, axis=0)
    x_axis = []
    y_axis = []
    wx_axis = []
    wy_axis = []

    for i in range(len(x)):
        x_axis.append(x[i][0])
        y_axis.append(x[i][1])
        # wx_axis.append(w[winner[i]][0])
        # wy_axis.append(w[winner[i]][1])
        wx_axis.append(w[i][0])
        wy_axis.append(w[i][1])
    x_axis.extend(wx_axis)
    y_axis.extend(wy_axis)
    plt.scatter(x_axis, y_axis)
    # print(x_axis, y_axis)

    for i in range(10):
        plt.annotate(str(i), xy=(x_axis[i], y_axis[i]), xytext=(
            x_axis[i] + 15., y_axis[i] + 15.))  # 这里xy是需要标记的坐标，xytext是对应的标签坐标

    for i in range(10, 20):    # 权重的数据
        plt.annotate(('w' + str(i - 10)), xy=(x_axis[i], y_axis[i]), xytext=(
            x_axis[i] - 16.4, y_axis[i] - 16.4))  # 这里xy是需要标记的坐标，xytext是对应的标签坐标

    # 再将winner的外星向量样本连接起来
    x_axis = []
    y_axis = []
    for i in range(len(winner2)):
        x_axis.append(w[winner2[i]][0])
        y_axis.append(w[winner2[i]][1])
    plt.plot(x_axis, y_axis)
    plt.show()
