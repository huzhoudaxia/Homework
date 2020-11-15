# coding=UTF-8
import numpy as np
import math
import random  # np的np.random.randint(0, 25, 5)可能生成相等的随机数
import matplotlib.pyplot as plt
import ast
from prettytable import PrettyTable
from tool import tool


if __name__ == '__main__':
    #  题目三
    tool = tool()
    data, lable = tool.read_txt()
    for i in range(len(data)):
        # print(lable[i], i)
        test = abs(np.array(data[i]) * 255 - 255)
        # print(test)
        test = np.reshape(test, [9, 7])

        plt.imshow(test, cmap='gray')
        plt.title(lable[i])
        # plt.show()

    x = np.array(data)
    w = np.array([np.random.rand(63) for i in range(25)])
    tool.dot2 = (100000., 0.01)
    tool.dot1 = (0., 0.6)
    tool.iteration = 100000

    # 归一化
    input_norm = tool.Normalize_new(x)
    w_norm = tool.Normalize_new(w)

    # 训练
    for ite in range(tool.iteration):
        for i in range(len(x)):
            dis = tool.Euclidean(x[i], w)
            data, inde = tool.find_winner(dis)
            w[inde] = tool.WTA(x[i], w[inde], 0, ite)
            # w = w_norm
            # w_norm = tool.Normalize_new(w)

    win_lable = []
    result = np.empty(shape=(0))

    for i in range(len(x)):
        dis = tool.Euclidean(x[i], w)
        data, inde = tool.find_winner(dis)
        result = np.append(result, [inde], axis=0)
        win_lable.append(lable[i])  # 进而找出对应的class
    clas = tool.lable_gather(win_lable, result)
    '''
    winner = []
    win_lable = []
    for i in range(len(x)):
        out = tool.neuron(x[i], w)
        winner.append(list(out).index(max(out)))
        win_lable.append(lable[i])  # 进而找出对应的class

    clas = tool.lable_gather(win_lable, winner)
    '''
    print(clas)
