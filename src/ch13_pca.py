# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 21:56:28 2017
主成分分析模板
@author: Burning
"""

import numpy
from matplotlib import pyplot as plt
from numpy import mat, shape, mean, cov, linalg, argsort, nonzero, isnan, sum

def 加载数据集(文件路径, 分隔符='\t'):
    数据集 = []
    文件 = open(文件路径)
    for 行 in 文件.readlines():
        数据集.append(list(map(float, 行.strip().split(分隔符))))
    文件.close()
    return mat(数据集)

def 主成分分析(数据矩阵, 最大特征数=9999999):
    均值 = mean(数据矩阵, axis=0)
    去均值数据 = 数据矩阵 - 均值
    协方差矩阵 = cov(去均值数据, rowvar=False)#去均值并不影响协方差计算
    特征值, 特征向量 = linalg.eig(协方差矩阵)
    特征值索引 = argsort(特征值)
    主成分索引 = 特征值索引[:-最大特征数-1:-1]#双冒号表示[起点:终点:步长]
    有序特征向量 = 特征向量[:,主成分索引]
    低维数据矩阵 = 去均值数据 * 有序特征向量
    重构数据矩阵 = (低维数据矩阵 * 有序特征向量.T) + 均值
    return 低维数据矩阵, 重构数据矩阵

plt.rcParams['font.sans-serif']=['SimHei']#设置中文字体

数据矩阵 = 加载数据集('../data/ch13_testSet.txt')
低维数据, 重构数据 = 主成分分析(数据矩阵, 1)
图像 = plt.subplot(111)
图像.scatter(数据矩阵[:,0].flatten().A[0], 数据矩阵[:,1].flatten().A[0], marker='^', s=9)
图像.scatter(重构数据[:,0].flatten().A[0], 重构数据[:,1].flatten().A[0], marker='o', s=5, c='red')
plt.show()

数据矩阵 = 加载数据集('../data/ch13_secom.data', ' ')
for i in range(shape(数据矩阵)[1]):
    均值 = mean(数据矩阵[nonzero(~isnan(数据矩阵[:,i].A))[0], i])
    数据矩阵[nonzero(isnan(数据矩阵[:,i].A))[0], i] = 均值
协方差矩阵 = cov(数据矩阵, rowvar=False)
特征值, 特征向量 = linalg.eig(协方差矩阵)
特征索引 = argsort(特征值)[::-1]
有序特征值 = 特征值[特征索引]
总方差 = sum(有序特征值)
方差占比 = 有序特征值 / 总方差 * 100
绘图维数 = 20
图像 = plt.subplot(111)
图像.plot(range(1,绘图维数+1), 方差占比[:绘图维数], marker='^')
plt.xlabel('主成分次序')
plt.ylabel('方差占比%')