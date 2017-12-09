# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 12:12:57 2017
逻辑回归及主要优化方法模板
@author: Burning
"""

import numpy
import matplotlib
from numpy import shape, ones

def 加载数据集(文件路径):
    from numpy import mat
    特征 = []
    标签 = []
    文件 = open(文件路径)
    for 行 in 文件.readlines():
        特征.append([float(元素) for 元素 in 行.strip().split('\t')[:-1]])
        #原作中这里多加了个1，理解为在随机梯度下降中作为L1正则惩罚项
        标签.append(float(行.strip().split('\t')[-1]))
    文件.close()
    return mat(特征), mat(标签).T

def S形函数(x):
    from numpy import exp
    return 1 / (1 + exp(-x))

def 完全迭代梯度提升(特征矩阵, 标签矩阵, 学习率=0.001, 最大迭代次数=500):
    条目数, 特征数 = shape(特征矩阵)
    权重 = ones((特征数, 1))
    for i in range(最大迭代次数):
        预测 = S形函数(特征矩阵 * 权重)
        损失 = 标签矩阵 - 预测
        权重 += 学习率 * 特征矩阵.T * 损失
        if abs(损失.sum()) < 0.5:
            break
    return 权重

def 随机顺序梯度上升(特征矩阵, 标签矩阵, 学习率=0.01):
    条目数, 特征数 = shape(特征矩阵)
    权重 = ones((特征数, 1))
    for i in range(条目数):
        预测 = S形函数(特征矩阵[i] * 权重)
        损失 = 标签矩阵[i] - 预测
        权重 += 学习率 * 特征矩阵[i].T * 损失
    return 权重

def 随机渐进梯度上升(特征矩阵, 标签矩阵, 迭代次数=150):
    from numpy.random import randint#numpy的randint和random的用法不一样
    条目数, 特征数 = shape(特征矩阵)
    权重 = ones((特征数, 1))
    for j in range(迭代次数):
        数据索引 = list(range(条目数))
        for i in range(条目数):
            学习率 = 4 / (1.0 + j + i)
            随机索引 = randint(0, len(数据索引))
            预测 = S形函数(特征矩阵[随机索引] * 权重)
            损失 = 标签矩阵[随机索引] - 预测
            权重 = 权重 + 学习率 * float(损失) * 特征矩阵[随机索引].T
            del(数据索引[随机索引])
    return 权重

def 绘制二维二分类效果(特征矩阵, 标签矩阵, 权重):
    from matplotlib import pyplot as plt
    from numpy import mat, arange
    x = arange(-4, 3, 0.1)
    y = (1 + 权重[0] * x) / -权重[1]#s(wx)=1/(1+e^(-wx))=0.5 => w0x0+w1x1=-1
    plt.rcParams['font.sans-serif']=['SimHei']#设置中文字体
    #窗口 = plt.figure()
    图像 = plt.subplot(111)
    图像.scatter(特征矩阵[:,0][标签矩阵==0].reshape(-1).tolist(), 特征矩阵[:,1][标签矩阵==0].reshape(-1).tolist(), s=30, c='red')
    图像.scatter(特征矩阵[:,0][标签矩阵==1].reshape(-1).tolist(), 特征矩阵[:,1][标签矩阵==1].reshape(-1).tolist(), s=30, c='green')
    图像.plot(x, y)
    plt.xlabel('特征0')
    plt.ylabel('特征1')
    plt.show()

def S形函数分类(特征, 权重):
    return S形函数(特征 * 权重) > 0.5

def 疝气数据集测试(测试次数=2):
    from numpy import size
    总错误率 = 0
    for i in range(测试次数):
        训练特征, 训练标签 = 加载数据集('../data/ch05_horseColicTraining.txt')
        权重 = 随机渐进梯度上升(训练特征, 训练标签)
        测试特征, 测试标签 = 加载数据集('../data/ch05_horseColicTest.txt')
        预测 = S形函数分类(测试特征, 权重)
        错误率 = sum(预测 != 测试标签) / size(测试标签)
        print('第%d次错误率:'%i, 错误率)
        总错误率 += 错误率
    print('平均错误率:', 总错误率/测试次数)

特征矩阵, 标签矩阵 = 加载数据集('../data/ch05_testSet.txt')
权重 = 完全迭代梯度提升(特征矩阵, 标签矩阵)
绘制二维二分类效果(特征矩阵, 标签矩阵, numpy.array(权重))
疝气数据集测试()