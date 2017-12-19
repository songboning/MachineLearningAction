# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 15:51:35 2017
基本回归类算法模板,标签使用列矢量
@author: Burning
"""

import numpy
import matplotlib
from numpy import mat

def 加载数据集(文件路径):
    特征 = []
    标签 = []
    文件 = open(文件路径)
    for 行 in 文件.readlines():
        特征.append([float(元素) for 元素 in 行.strip().split('\t')[:-1]])
        标签.append(float(行.strip().split('\t')[-1]))
    文件.close()
    return mat(特征), mat(标签).T

def 绘图(特征矩阵, 标签矩阵, 有序特征, 有序预测):
    from matplotlib import pyplot as plt
    图像 = plt.subplot(111)
    图像.scatter(特征矩阵[:,1].flatten().A[0], 标签矩阵.flatten().A[0], s=5, c='red')
    图像.plot(有序特征[:,1], 有序预测)#按列矢量连线
    plt.show()
#针对样例中第一维恒为1的情况

def 标准线性回归(特征矩阵, 标签矩阵):
    from numpy import linalg
    导数中间量 = 特征矩阵.T * 特征矩阵
    if not linalg.det(导数中间量):
        print ('矩阵奇异,不可逆')
        return
    回归系数 = 导数中间量.I * (特征矩阵.T * 标签矩阵)
    return 回归系数
#损失函数=求和(预测值-实际值)^2
#损失函数导数为0时W=(X.T*X)^-1*X.T*Y

def 局部加权线性回归(特征矩阵, 标签矩阵, 预测点, 距离衰减系数k=1):
    from numpy import shape, eye, exp, linalg
    点数 = shape(特征矩阵)[0]
    权重 = mat(eye(点数))
    for i in range(点数):
        各项距离矩阵 = 预测点 - 特征矩阵[i,:]
        权重[i,i] = exp(各项距离矩阵 * 各项距离矩阵.T / (-2.0 * 距离衰减系数k ** 2))
    导数中间量 = 特征矩阵.T * (权重 * 特征矩阵)
    if not linalg.det(导数中间量):
        print ('矩阵奇异,不可逆')
        return
    回归系数 = 导数中间量.I * (特征矩阵.T * (权重 * 标签矩阵))
    预测值 = 预测点 * 回归系数
    return 预测值
#损失函数=求和[(预测值-实际值)*权值]^2
#这里权值使用随着距预测点远离而高斯衰减
#本函数每次预测一个点
#损失函数导数为0时W=(X.T*权重*X)^-1*X.T*权重*Y

def 局部加权线性回归测试(特征矩阵, 标签矩阵, 距离衰减系数k=1):
    from numpy import shape
    点数 = shape(特征矩阵)[0]
    预测矩阵 = 标签矩阵.copy()
    for i in range(点数):
        预测矩阵[i] = 局部加权线性回归(特征矩阵, 标签矩阵, 特征矩阵[i,:], 距离衰减系数k)
    return 预测矩阵

def 均方误差(标签, 预测):
    return ((标签 - 预测) ** 2).sum()

def 归一化矩阵(数据矩阵):
    from numpy import mean, var
    均值 = mean(数据矩阵, 0)
    方差 = var(数据矩阵, 0)
    归一化数据 = (数据矩阵 - 均值) / 方差
    return 归一化数据

def 岭回归(特征矩阵, 标签矩阵, 岭参数=0.2):
    from numpy import shape, eye, linalg
    导数中间量 = 特征矩阵.T * 特征矩阵 + 岭参数 * eye(shape(特征矩阵)[1])
    if not linalg.det(导数中间量):
        print ('矩阵奇异,不可逆')
        return
    回归系数 = 导数中间量.I * (特征矩阵.T * 标签矩阵)
    return 回归系数
#损失函数=求和(预测值-实际值)^2
#损失函数导数为0时W=(X.T*X+lamda*I)^-1*X.T*Y
#加入岭使得再特征数大于样本数时,导数中间量仍然可逆

def 岭回归测试(特征矩阵, 标签矩阵, 测试数=30):
    from numpy import zeros, shape, exp
    from matplotlib import pyplot as plt
    归一化特征 = 归一化矩阵(特征矩阵)
    归一化标签 = 归一化矩阵(标签矩阵)
    回归系数矩阵 = zeros((测试数, shape(特征矩阵)[1]))
    for i in range(测试数):
        回归系数 = 岭回归(归一化特征, 归一化标签, exp(i-10))
        回归系数矩阵[i,:] = 回归系数.T
    图像 = plt.subplot(111)
    图像.plot(回归系数矩阵)
    plt.show()
    return 回归系数矩阵

def 逐步向前线性回归(特征矩阵, 标签矩阵, 步长=0.01, 迭代次数=100):
    from numpy import mean, shape, inf, zeros
    from matplotlib import pyplot as plt
    归一化特征 = 归一化矩阵(特征矩阵)
    标签 = 标签矩阵 - mean(标签矩阵)
    特征数 = shape(特征矩阵)[1]
    回归系数矩阵 = zeros((迭代次数, shape(特征矩阵)[1]))
    回归系数 = zeros((特征数, 1))
    for i in range(迭代次数):
        最小误差 = inf
        for j in range(特征数):
            for 方向 in [-1, 1]:
                回归系数测试 = 回归系数.copy()
                回归系数测试[j] += 步长 * 方向
                预测 = 归一化特征 * 回归系数测试
                误差 = 均方误差(标签.A, 预测.A)
                if 误差 < 最小误差:
                    最小误差 = 误差
                    最佳回归系数 = 回归系数测试.copy()
        回归系数 = 最佳回归系数.copy()
        回归系数矩阵[i,:] = 回归系数.T
    图像 = plt.subplot(111)
    图像.plot(回归系数矩阵)
    plt.show()
    return 回归系数

特征矩阵, 标签矩阵 = 加载数据集('../data/ch08_ex0.txt')
回归系数 = 标准线性回归(特征矩阵, 标签矩阵)
相关系数 = numpy.corrcoef((特征矩阵*回归系数).T, 标签矩阵.T)#行与行之间的相关系数
顺序索引 = 特征矩阵[:,1].argsort(0)
有序特征 = 特征矩阵[顺序索引][:,0,:]#索引完形状变成三维了
#两行有序预测二选一即可测试标准线性回归和局部加权线性回归
#有序预测 = 有序特征 * 回归系数
有序预测 = 局部加权线性回归测试(特征矩阵, 标签矩阵, 距离衰减系数k=0.01)[顺序索引][:,0,:]
绘图(特征矩阵, 标签矩阵, 有序特征, 有序预测)

特征矩阵, 标签矩阵 = 加载数据集('../data/ch08_abalone.txt')
#岭回归测试(特征矩阵, 标签矩阵)
逐步向前线性回归(特征矩阵, 标签矩阵, 0.005, 1000)