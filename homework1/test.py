import numpy as np
import pandas as pd
from prettytable import PrettyTable
import ast
# df_news = pd.read_table('ABCDEFGHIJ.txt', header=None)
# print(df_news[0])
xdata = np.empty(shape=(0, 63))
a = np.array([[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0,
               0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]])
xdata = np.append(xdata, a, axis=0)
xdata = np.append(xdata, a, axis=0)


num1 = 500
num2 = 200
iteration = 50000
data1 = np.random.uniform(low=-4, high=-1, size=num1)
data2 = np.random.uniform(low=-1, high=4, size=num2)
xdata = np.empty(shape=(0))
xdata = np.append(xdata, data1, axis=0)
xdata = np.append(xdata, data2, axis=0)
xdata.sort()
print(xdata)
