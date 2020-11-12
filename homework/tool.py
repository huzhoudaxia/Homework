# coding=UTF-8
import numpy as np
import math
import random  # np的np.random.randint(0, 25, 5)可能生成相等的随机数
import matplotlib.pyplot as plt
import ast
from prettytable import PrettyTable

dot2 = (300., 0.0)
dot1 = (0., 0.6)
iteration = 300


class tool():
    def __init__(self):
        self.dot2 = (300., 0.0)
        self.dot1 = (0., 0.6)
        self.iteration = 300

    def Euclidean(self, x, w):
        # :: x为单个输入向量，数据结构，ndarray,[input_train]
        # :: w为权重向量，数据结构：ndarray,[w1,w2,...]
        # :: x与y为数组乘积，sqrt(matmul(x,w))
        # :: 输入数据是归一化后的
        # :: 计算每组输入与神经元的距离，返回值shape:len(x)*len(w)数据类型为list
        dis = []
        for i in range(len(w)):
            tmp = x - w[i]
            dis.append(np.sqrt(np.inner(tmp, tmp)))
        return dis

    def find_winner(self, dis):
        # :: 输入数据为list,一维
        # :: function,寻找一维list中最小的数，并且返回其最小值的索引
        dis = list(dis)
        min_data = min(dis)
        inde = dis.index(min_data)
        return min_data, inde

    def cal_fomular(self, dot1, dot2):
        # :: 两点坐标必须是浮点数
        # ::
        k = float(dot2[1] - dot1[1]) / float(dot2[0] - dot1[0])
        b = float(dot1[1] - k * dot1[0])
        return k, b

    def get_learn_rate(self, N, t):
        # :: N拓扑结构距离
        # :: t训练次数
        # :: 有可能是分段函数，迭代超过T——th的话会截胡
        k, b = self.cal_fomular(self.dot1, self.dot2)     # 线性部分
        if(t <= iteration):
            l_nt = (k * t + b) * math.pow(math.e, -N)
        else:
            l_nt = dot2[1]
        return l_nt

    def out_WTA(self, dout, out, w, N, t):
        '''
        # :: w为竞争层获胜神经元与输出神经元的权重.数据结构shape:实数
        # :: dout为输入向量,表示期望输出，数据结构为ndarray，shape：实数
        # :: function为调整权重
        # :: 返回的权重结构：实数
        '''
        l_nt = self.get_learn_rate(N, t)
        # l_nt = (0.5 - 0.05 * t) * math.pow(math.e, -N)
        b = w + l_nt * (dout - out)
        # print(w)
        return b

    def WTA(self, x, w, N, t):
        # :: w为获胜神经元的权值矩阵.数据结构为ndarray，shape:1维
        # :: x为输入向量，数据结构为ndarray,shape:1维
        # :: function为调整权重
        # :: 返回的权重仍是ndarray,shape:1维
        l_nt = self.get_learn_rate(N, t)
        # l_nt = (0.5 - 0.05 * t) * math.pow(math.e, -N)
        # print(l_nt)
        w = w + l_nt * (x - w)
        return w

    def neuron_act(self, x, w):
        '''
        # :: x表示一组输入数据,一维ndarray矩阵
        # :: w表示所有输出神经元的输出矩阵，二维矩阵
        # :: 输出out表示一组输出神经元的输出向量,数据类型:ndarray，1维
        param hout:相当于激活函数的输出，获胜者输出1，败者输出0
        '''
        dis = self.Euclidean(x, w)  # 确定竞争层竞争获胜神经元
        data, inde = self.find_winner(dis)
        hout = np.array([0 for i in range(len(w))])
        hout[inde] = 1
        return hout

    def neuron(self, x, w):
        '''
        # :: x表示一组输入数据,一维ndarray矩阵
        # :: w表示所有输出神经元的输出矩阵，二维矩阵
        # :: 输出out表示一组输出神经元的输出向量,数据类型:ndarray，1维
        param hout:相当于激活函数的输出，获胜者输出1，败者输出0
        '''
        out = np.inner(x, w)
        out = np.round(out, 5)

        return out

    def add_noise(self, data, inde):
        # :: data数据结构为ndarray,sam1
        # :: function产生sam的1个噪声样本
        # :: 返回ndarray结构数据，1维结构
        data[inde] = abs(data[inde] - 1)
        return data

    def add_noise_all(self, data):
        # :: data数据结构为ndarray,sam1
        # :: function产生sam的所有噪声样本
        # :: 返回ndarray，二维结构，元素为ndarray
        noise = np.empty(shape=(0, 25))
        for i in range(len(data)):
            tmp = self.add_noise(data, i)
            noise = np.append(noise, [tmp], axis=0)
        return noise

    def make_samples(self, x):
        # :: x为正确的一组输入样本，数据结构ndarray，[x1,x2,...]
        # :: function为产生这组样本的噪声样本
        # :: 返回值用ndarray，shape:3,25,25
        noise = np.empty(shape=(0, 25, 25))  # 最终结构应该是一个三维矩阵，里面元素是每个样本的多个噪声样本
        for i in range(len(x)):
            data = x[i]  # 单个样本矩阵
            noise_tmp = self.add_noise_all(data)  # 二维矩阵ndarray
            noise = np.append(noise, [noise_tmp], axis=0)
        return noise

    def pic_train(self, noise):
        # :: noise为总的噪声样本，三维ndarray结构[sam1样本，Sam2样本...]
        # :: 返回值为三维结构ndarray，[sam1的5个样本，sam2的5个样本，sam3的5个样本...]
        small_sam = np.empty(shape=(0, 5, 25))
        pred_sam = np.empty(shape=(0, 20, 25))
        for i in range(len(noise)):  # 3类噪声样本
            noise_tmp = noise[i]  # 从中随机取5个噪声样本,二维结构ndarray
            noise_tmp2 = np.empty(shape=(0, 25))  # train的tmp
            inde = random.sample(range(0, 25), 5)
            # print(len(inde))
            for j in range(len(inde)):
                noise_tmp2 = np.append(
                    noise_tmp2, [noise_tmp[inde[j]]], axis=0)
            noise_tmp3 = np.delete(noise_tmp, inde, axis=0)
            pred_sam = np.append(pred_sam, [noise_tmp3], axis=0)
            small_sam = np.append(small_sam, [noise_tmp2], axis=0)
        return small_sam, pred_sam

    def lable_class(self, lable, num=5):
        # :: function,将同一类别的index记录在同一类中
        clas = {}
        for i in range(len(lable)):
            clas[str(lable[i])] = []
        for i in range(len(lable)):
            clas[str(lable[i])].append(i)
        return clas

    def lable_gather(self, lable, winner):
        '''
        :param function,将同一类别的index记录在同一类中
        :param clas:key为lable，value为位置
        '''
        clas = {}
        for i in range(len(lable)):
            clas[str(lable[i])] = []
        for i in range(len(winner)):
            clas[str(lable[i])].append(winner[i])
        return clas

    def table_show(self, data):
        '''将一组聚类数据在表格中展示
        :param data:dict数据
        :有问题
        '''
        table = [[] for i in range(25)]
        for key, value in data.items():
            for i in range(len(value)):
                table[value[i]] = key
        print(table)
        x = PrettyTable()
        for i in range(5):
            x.add_row([table[0 + (i * 5)], table[1 + (i * 5)],
                       table[2 + (i * 5)], table[3 + (i * 5)], table[4 + (i * 5)]])
        print(x)

    def read_txt(self):
        #  文件，
        #  逐行读取
        f = open('./Char7Data.txt')
        data = f.readlines()  # 逐行读取txt并存成list。每行是list的一个元素，数据类型为str
        dot = []
        lable = []
        for i in range(len(data)):  # len(data)为数据行数
            # :: i为偶数表示点阵数据，i为奇数表示标签数据
            # l.append(data[i].split(' ')[j])
            if i % 2 == 0:  # 偶数
                dot.append(ast.literal_eval(data[i]))
            else:
                lable.append(ast.literal_eval(data[i]))
        for i in range(len(lable)):
            if(lable[i] == [1, 0, 0, 0, 0, 0, 0]):
                lable[i] = 'A'
            elif(lable[i] == [0, 1, 0, 0, 0, 0, 0]):
                lable[i] = 'B'
            elif(lable[i] == [0, 0, 1, 0, 0, 0, 0]):
                lable[i] = 'C'
            elif(lable[i] == [0, 0, 0, 0, 0, 0, 1]):
                lable[i] = 'K'
            elif(lable[i] == [0, 0, 0, 0, 0, 1, 0]):
                lable[i] = 'J'
            elif(lable[i] == [0, 0, 0, 0, 1, 0, 0]):
                lable[i] = 'E'
            elif(lable[i] == [0, 0, 0, 1, 0, 0, 0]):
                lable[i] = 'D'
        return dot, lable

    def Normalize(self, data):
        '''
        param X` = X / ||X||
        param data,type:ndarray,shape:不限

        '''
        num = np.sqrt(np.sum(data ** 2))
        data = data.astype(np.float)

        dat = data / float(num)
        return dat

    def Normalize_new(self, data):
        '''
        param X`i = Xi / ||Xi||,i=1,2,...
        param data,type:ndarray,shape:二维

        '''
        data = data.astype(np.float)
        dat = np.empty(shape=(0, len(data[0])))
        for i in range(len(data)):
            num = np.sqrt(np.sum(data[i] ** 2))
            dat = np.append(dat, [data[i] / float(num)], axis=0)
        return dat


if __name__ == '__main__':
    a = np.array([2])
    b = np.empty(shape=(0))
    b = np.append(b, a, axis=0)
    b = np.append(b, a, axis=0)
    print(b)
    '''
    # hmw2
    c_sam = np.array([0, 1, 1, 1, 0,
                      1, 0, 0, 0, 1,
                      1, 0, 0, 0, 0,
                      1, 0, 0, 0, 1,
                      0, 1, 1, 1, 0])
    i_sam = np.array([0, 1, 1, 1, 0,
                      0, 0, 1, 0, 0,
                      0, 0, 1, 0, 0,
                      0, 0, 1, 0, 0,
                      0, 1, 1, 1, 0])
    t_sam = np.array([1, 1, 1, 1, 1,
                      0, 0, 1, 0, 0,
                      0, 0, 1, 0, 0,
                      0, 0, 1, 0, 0,
                      0, 0, 1, 0, 0, ])
    x = np.array([c_sam, i_sam, t_sam])
    w = np.array([np.random.rand(25) for i in range(3)])
    noise_sam = make_samples(x)
    small_sam, pred_sam = pic_train(noise_sam)
    small_sam = np.insert(small_sam, 0, [[c_sam, i_sam, t_sam]], axis=1)
    print(small_sam.shape)
    small_sam = np.reshape(small_sam, (18, 25))
    pred_sam = np.reshape(pred_sam, (60, 25))
    # 训练

    for i in range(len(x)):
        dis = Euclidean(x[i], w)
        data, inde = find_winner(dis)
        w[inde] = WTA(x[i], w[inde], 1, 1)
    # 识别
    test = w[0] * 255
    test = np.reshape(test, [5, 5])
    # plt.imshow(test, cmap='Greys_r')
    # plt.show()

    #  题目三
    data, lable = read_txt()
    for i in range(len(data)):
        # print(lable[i], i)
        test = abs(np.array(data[i]) * 255 - 255)
        # print(test)
        test = np.reshape(test, [9, 7])

        plt.imshow(test, cmap='gray')
        plt.title(lable[i])
        plt.show()

    x = np.array(data)
    w = np.array([np.random.rand(63) for i in range(25)])
    for ite in range(iteration):
        for i in range(len(x)):
            dis = Euclidean(x[i], w)
            data, inde = find_winner(dis)
            w[inde] = WTA(x[i], w[inde], 0, ite)
    winner = []
    win_lable = []
    for i in range(len(x)):
        out = neuron(x[i], w)
        winner.append(list(out).index(max(out)))
        win_lable.append(lable[i])  # 进而找出对应的class

    clas = lable_gather(win_lable, winner)
    # 表格显示

    # 题目6
    x = np.array([[246, 53], [408, 79], [909, 89], [115, 264], [396, 335], [
                 185, 456], [699, 252], [963, 317], [922, 389], [649, 515]])
    w = np.array([np.random.randint(100, 200, 2) for i in range(10)])
    for ite in range(iteration):
        for i in range(len(x)):
            tmp = Normalize_one(x[i])
            tmp_w = Normalize_two(w)
            dis = Euclidean(tmp, tmp_w)
            data, inde = find_winner(dis)
            if(inde > 0 and inde < len(w) - 1):
                w[inde] = WTA(x[i], w[inde], 0, ite)
                w[inde - 1] = WTA(x[i], w[inde - 1], 1, ite)
                w[inde + 1] = WTA(x[i], w[inde + 1], 1, ite)
            elif(inde == 0):
                w[inde] = WTA(x[i], w[inde], 1, ite)
                w[len(w) - 1] = WTA(x[i], w[len(w) - 1], 1, ite)
            elif(inde == len(w) - 1):
                w[inde] = WTA(x[i], w[inde], 1, ite)
                w[0] = WTA(x[i], w[0], 1, ite)
    for ite in range(iteration):
        for i in range(len(x)):
            tmp = Normalize_one(x[i])
            tmp_w = Normalize_two(w)
            dis = Euclidean(tmp, tmp_w)
            data, inde = find_winner(dis)
            w[inde] = WTA(x[i], w[inde], 0, ite)

    winner = []
    for i in range(len(x)):
        out = neuron(x[i], w)
        winner.append(list(out).index(max(out)))
    print(winner)
    '''
    '''
    # 题目7
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
    x_norm = Normalize(xdata)
    for ite in range(iteration):  # 第一阶段
        for i in range(len(xdata)):

            tmp_w = Normalize(w)
            dis = Euclidean(x_norm[i], tmp_w)
            data, inde = find_winner(dis)
            w[inde] = WTA(xdata[i], w[inde], 0, ite)
    # print(w)
    dot2 = (300., 0.0)
    dot1 = (0., 0.1)
    iteration = 300
    # 调整第二阶段的学习率
    # np.random.shuffle(xdata)
    for ite in range(iteration):    # 第二阶段调整外星向量
        for i in range(len(xdata)):
            hout = neuron(xdata[i], w)
            yout = neuron(hout, w_out)
            data, inde = find_winner(yout)
            # print(inde)
            print(ydata[i], hout, yout, w_out[inde])
            w_out[inde] = out_WTA(ydata[i], yout, w_out[inde], 0, ite)
            print(w_out)

    '''
    '''
    hout = neuron(xdata[0], w)
    yout = neuron(hout, w_out)
    data, inde = find_winner(yout)
    # print(inde)
    print(ydata[0], yout, w_out[inde])
    w_out[inde] = out_WTA(ydata[0], yout, w_out[inde], 0, 1)
    print(w_out)
    hout = neuron(xdata[1], w)
    yout = neuron(hout, w_out)
    data, inde = find_winner(yout)
    # print(inde)
    print(ydata[1], yout, w_out[inde])
    w_out[inde] = out_WTA(ydata[1], yout, w_out[inde], 0, 1)
    print(w_out)
    '''
