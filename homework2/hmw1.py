# coding=UTF-8
import numpy as np
from tool import tool
if __name__ == '__main__':
    # hmw1
    tool = tool()
    x = np.array([[0.707, 0.707], [0, 1], [-0.643, 0.766],
                  [-1, 0], [-0.707, -0.707]])
    # w1=[1,0],w2=[0,-1]
    w = np.array([[1., 0.], [0., -1.]])  # 这种形式采用矩阵内积
    tool.iteration = 0
    tool.dot1 = (0, 0.6)   # 描述学习率函数
    # 归一化
    input_nor = np.empty(shape=(0, len(x[0])))
    for i in range(len(x)):
        input_nor = np.append(input_nor, [tool.Normalize(x[i])], axis=0)
    w_nor = np.empty(shape=(0, len(w)))
    for i in range(len(w[0])):
        w_nor = np.append(w_nor, [tool.Normalize(w[i])], axis=0)

    for i in range(len(x)):
        dis = tool.Euclidean(input_nor[i], w_nor)
        data, inde = tool.find_winner(dis)
        w[inde] = tool.WTA(x[i], w[inde], 0, 0)
    print(w)
