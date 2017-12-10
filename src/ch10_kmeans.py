# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 22:45:38 2017
K均值聚类模板
@author: Burning
"""

import numpy
import matplotlib
from numpy import shape, mat, zeros

def 加载数据集(文件路径):
    数据集 = []
    文件 = open(文件路径)
    for 行 in 文件.readlines():
        数据集.append(list(map(float, 行.strip().split('\t'))))
    文件.close()
    return mat(数据集)

def 绘制二维数据点和簇心(数据矩阵, 簇心矩阵):
    from matplotlib import pyplot as plt
    图像 = plt.subplot(111)
    图像.scatter(数据矩阵[:,0].reshape(-1).tolist(), 数据矩阵[:,1].reshape(-1).tolist(), s=30, c='blue')
    图像.scatter(簇心矩阵[:,0].reshape(-1).tolist(), 簇心矩阵[:,1].reshape(-1).tolist(), s=50, c='red', marker='+')
    plt.show()

def 计算欧拉距离(点1, 点2):
    from numpy import sqrt, power, sum
    return sqrt(sum(power(点1 - 点2, 2)))

def 随机k个中心(数据矩阵, k):
    from numpy import random
    维度 = shape(数据矩阵)[1]
    中心坐标矩阵 = mat(zeros((k, 维度)))
    for i in range(维度):
        最小值 = min(数据矩阵[:, i])
        范围 = max(数据矩阵[:, i] - 最小值)
        中心坐标矩阵[:, i] = mat(最小值 + random.rand(k, 1) * 范围)
    return 中心坐标矩阵

def k均值聚类(数据矩阵, k, 距离计算函数=计算欧拉距离, 随机中心函数=随机k个中心):
    from numpy import mean, inf, nonzero
    点数 = shape(数据矩阵)[0]
    簇分配 = mat(zeros((点数, 1))) - 1
    簇中心 = 随机中心函数(数据矩阵, k)
    簇分配发生变化 = True
    while 簇分配发生变化:
        簇分配发生变化 = False
        for i in range(点数):
            最近距离 = inf
            最近簇号 = -1
            for j in range(k):
                点i到簇j距离 = 距离计算函数(数据矩阵[i,:], 簇中心[j,:])
                if 点i到簇j距离 < 最近距离:
                    最近距离 = 点i到簇j距离
                    最近簇号 = j
            if 簇分配[i,0] != 最近簇号:
                簇分配发生变化 = True
                簇分配[i,0] = 最近簇号
        #print (簇中心)
        for 簇号 in range(k):
            簇中点 = 数据矩阵[nonzero(簇分配[:,0].A == 簇号)[0]]
            簇中心[簇号, :] = mean(簇中点, axis=0)
    return 簇中心, 簇分配

def 二分k均值聚类(数据矩阵, k, 距离计算函数=计算欧拉距离, 随机中心函数=随机k个中心):
    from numpy import mean, inf, nonzero, column_stack
    点数 = shape(数据矩阵)[0]
    簇分配及代价 = mat(zeros((点数, 2)))#这里还表示把所有点默认分配给起点
    起点 = mean(数据矩阵, axis=0).tolist()[0]
    簇中心列表 = [起点]
    for i in range(点数):
        簇分配及代价[i,1] = 距离计算函数(起点, 数据矩阵[i,:]) ** 2
    while len(簇中心列表) < k:
        最小误差平方和 = inf
        for 簇号 in range(len(簇中心列表)):
            簇中点 = 数据矩阵[nonzero(簇分配及代价[:,0].A == 簇号)[0]]
            簇中心, 簇划分 = k均值聚类(簇中点, 2, 距离计算函数, 随机中心函数)
            划分代价 = [距离计算函数(簇中心[int(簇划分[i, 0])], 簇中点[i]) for i in range(shape(簇中点)[0])]
            划分部分总代价 = sum(划分代价)
            保留部分总代价 = sum(簇分配及代价[nonzero(簇分配及代价[:,0].A != 簇号)[0],1])
            if 划分部分总代价 + 保留部分总代价 < 最小误差平方和:
                最佳切分簇号 = 簇号
                最佳新簇中心 = 簇中心
                最佳切分分配 = 簇划分.copy()
                最佳切分代价 = mat(划分代价).T
                最小误差平方和 = 划分部分总代价 + 保留部分总代价
        最佳切分分配[nonzero(最佳切分分配[:,0].A == 1)[0], 0] = len(簇中心列表)#先分配新簇号,以免重复
        最佳切分分配[nonzero(最佳切分分配[:,0].A == 0)[0], 0] = 最佳切分簇号
        簇中心列表[最佳切分簇号] = 最佳新簇中心[0].tolist()[0]
        簇中心列表.append(最佳新簇中心[1].tolist()[0])
        簇分配及代价[nonzero(簇分配及代价[:,0].A == 最佳切分簇号)[0]]= column_stack((最佳切分分配, 最佳切分代价))
    return mat(簇中心列表), 簇分配及代价

数据矩阵 = 加载数据集('../data/ch10_testSet2.txt')
簇中心, 簇分配及代价 = 二分k均值聚类(数据矩阵, 3)
绘制二维数据点和簇心(数据矩阵, 簇中心)
#雅虎API用不了了
#感觉地理聚类意义不大以后在搞