# coding: utf8

import os

import matplotlib.pyplot as plt  # plt 用于显示图片
import matplotlib.image as mpimg  # mpimg 用于读取图片
import numpy as np


import cv2
import numpy as np


def local_threshold(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    # binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,25,10)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 10)
    cv2.imshow("binary1", binary)
    return binary


def extract_imglabel(dir):
    '''
    param function:从目录中提取图片数据和标签
    param label数据格式，一维ndarray，
    param pic：3维ndarray，单张图片数据格式为2维

    '''
    files = os.listdir(dir)
    label = np.empty(shape=(0))
    pic = np.empty(shape=(0, 40, 20))
    for name in files:
        label = np.append(label, [name[-5]], axis=0)
        img = cv2.imread(dir + '/' + name)

        tmp = local_threshold(img)
        resized = cv2.resize(tmp, (20, 40))  # 裁剪
        '''
        cv2.imshow("binary ", resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''
        # print(label)

        #pic = np.append(pic, [tmp], axis=0)
        break
    return pic, label


# extract_imglabel('./number/train')
'''

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


pic = mpimg.imread('./number/train/000001_P.jpg')  # 读出来即是nadarray格式

gray = rgb2gray(pic)
print(gray)
# 也可以用
# plt.imshow(gray, cmap=plt.get_cmap('gray'))
plt.imshow(gray, cmap='Greys_r')
plt.axis('off')
plt.show()

im_array = np.array(pic)


def extract_label(dir):
    files = os.listdir(dir)
    for name in files:
        print(name[-5])
        break
'''


def getUniqueItems(labels):
    label = []
    for item in labels:
        if item not in label:
            label.append(item)
    return label


a = ["asd", "def", "ase", "dfg", "asd", "def", "dfg"]
# print(getUniqueItems(a))
print(2**5)
a = np.array([[1, 2], [3, 4]])
b = a.flatten()
c = b.reshape((2, 2))
print(a, b, c)
