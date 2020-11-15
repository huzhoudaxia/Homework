# coding=UTF-8
import numpy as np
import math
import random  # np的np.random.randint(0, 25, 5)可能生成相等的随机数
import matplotlib.pyplot as plt
import ast
from prettytable import PrettyTable
from tool import tool


if __name__ == '__main__':
    tool = tool()
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
    '''
    w = np.array([[0.93584185, 0.23683706, 0.42223593, 0.59378865, 0.54272214, 0.40803855,
                   0.40564657, 0.84378821, 0.11471264, 0.1411507, 0.13720899, 0.93352089,
                   0.03781725, 0.82770237, 0.82693865, 0.75986563, 0.59656431, 0.50670765,
                   0.50099858, 0.93687296, 0.89485299, 0.08895751, 0.55357627, 0.54748098,
                   0.16058101],
                  [0.22134821, 0.87402254, 0.99488961, 0.85228134, 0.30468596, 0.77357613,
                   0.18244373, 0.68581541, 0.52418529, 0.79268011, 0.85263495, 0.04845165,
                   0.69266016, 0.85428395, 0.28120333, 0.47711909, 0.33135611, 0.353286,
                   0.55550303, 0.79701884, 0.64177523, 0.77261633, 0.98822408, 0.13337226,
                   0.44200107],
                  [0.10655718, 0.93786947, 0.68017278, 0.15212836, 0.97556877, 0.61430351,
                   0.58571519, 0.00725343, 0.35808762, 0.00901248, 0.57604472, 0.47820797,
                   0.11123438, 0.66184111, 0.68819169, 0.28630572, 0.82711379, 0.67185651,
                   0.45196806, 0.9190671, 0.95510386, 0.31780372, 0.16713319, 0.75681688,
                   0.42954655]])
    '''
    noise_sam = tool.make_samples(x)
    small_sam, pred_sam = tool.pic_train(noise_sam)
    # 将正确样本放在每类噪声样本的首位，数据结构3维
    small_sam = np.insert(small_sam, 0, [[c_sam, i_sam, t_sam]], axis=1)
    # 对数据结构进行重新组合，二维ndarray，shape：18 * 25
    small_sam = np.reshape(small_sam, (18, 25))
    pred_sam = np.reshape(pred_sam, (60, 25))    # 预测样本集
    # 训练
    # 归一化
    # input_norm = tool.Normalize_new(small_sam)

    # w_norm = tool.Normalize(w)

    tool.dot2 = (60000., 0.1)
    tool.dot1 = (0., 0.8)
    tool.iteration = 60000
    for ite in range(tool.iteration):
        for i in range(len(small_sam)):
            dis = tool.Euclidean(small_sam[i], w)
            data, inde = tool.find_winner(dis)
        # wta计算时用归一化的值还是原来值书上有两种表示
        # w = w_norm
        w[inde] = tool.WTA(small_sam[i], w[inde], 0, ite)

        # w_norm = tool.Normalize(w)
    print(w)
    # 看训练效果

    tool.dot2 = (1000., 0.0)
    tool.dot1 = (0., 0.1)
    tool.iteration = 1000
    for ite in range(tool.iteration):
        for i in range(len(small_sam)):
            dis = tool.Euclidean(small_sam[i], w)
            data, inde = tool.find_winner(dis)
        # wta计算时用归一化的值还是原来值书上有两种表示
        # w = w_norm
        w[inde] = tool.WTA(small_sam[i], w[inde], 0, ite)
        # w_norm = tool.Normalize(w)

    for i in range(len(w)):
        plt.subplot(221 + i)
        test = abs(np.array(w[i] * 255 - 255))
    # test = np.array(w[i] * 255)
        test = np.reshape(test, [5, 5])
        plt.imshow(test, cmap='Greys_r')
    plt.show()

    # 识别
    # 归一化
    # input_norm = tool.Normalize_new(pred_sam)
    result = np.empty(shape=(0))

    for i in range(len(pred_sam)):
        dis = tool.Euclidean(pred_sam[i], w)
        data, inde = tool.find_winner(dis)
        result = np.append(result, [inde], axis=0)
    result = np.reshape(result, (3, 20))
    # wta计算时用归一化的值还是原来值书上有两种表示
    print(result)
