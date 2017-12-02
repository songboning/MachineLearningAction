#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 20:29:51 2017
集成提升方法模版
@author: songboning
"""

import numpy
import matplotlib

def 载入小样例():
    from numpy import matrix
    特征 = matrix([[1., 2.1],
        [2. , 1.1],
        [1.3, 1. ],
        [1. , 1. ],
        [2. , 1. ]])
    标签 = [1.0, 1.0, -1.0, -1.0, 1.0]
    return 特征, 标签

def 决策树分类(特征矩阵, 划分维度, 阈值, 判定范围):
    from numpy import shape, ones
    分类结果 = ones((shape(特征矩阵)[0], 1))
    if 判定范围 == '小于':
        分类结果[特征矩阵[:,划分维度] <= 阈值] = -1.0
    else:# 判定范围 == '大于'
        分类结果[特征矩阵[:,划分维度] > 阈值] = -1.0
    return 分类结果

def 构建单层决策树(特征, 标签, 权重):
    from numpy import mat, shape, zeros, inf
    特征矩阵 = mat(特征)
    标签矩阵 = mat(标签).T
    行数, 列数 = shape(特征矩阵)
    步数 = 10
    最小错误值 = inf
    单层决策树 = dict()
    最佳预测值 = mat(zeros((行数,1)))
    for i in range(列数):
        最小值 = 特征矩阵[:,i].min()
        最大值 = 特征矩阵[:,i].max()
        步长 = (最大值 - 最小值) / 步数
        for j in range(-1, 步数+1):
            for 判定范围 in ['小于', '大于']:
                阈值 = 最小值 + j * 步长
                预测值 = 决策树分类(特征矩阵, i, 阈值, 判定范围)
                错误矩阵 = mat(zeros((行数, 1)))
                错误矩阵[预测值 != 标签矩阵] = 1
                错误权重 = 权重.T * 错误矩阵
                if 错误权重 < 最小错误值:
                    最小错误值 = 错误权重
                    单层决策树['划分维度'] = i
                    单层决策树['阈值'] = 阈值
                    单层决策树['判定范围'] = 判定范围
                    最佳预测值 = 预测值
    return 单层决策树, 最小错误值, 最佳预测值

def 训练ada提升(特征, 标签, 最大迭代次数=45):
    from numpy import mat, ones, zeros, log, multiply, exp, sign
    个数 = len(标签)
    权重 = mat(ones((个数, 1)) / 个数)
    弱分类器们 = []
    累计估计值 = mat(zeros((个数, 1)))
    for i in range(最大迭代次数):
        单层决策树, 错误值, 类别估计值 = 构建单层决策树(特征, 标签, 权重)
        #print('迭代%d权重:'%i, 权重)
        结果权重 = float(0.5 * log((1-错误值) / max(错误值, 1e-16)))
        单层决策树['结果权重'] = 结果权重
        弱分类器们.append(单层决策树)
        指数迭代量 = multiply(-1 * 结果权重 * mat(标签).T, 类别估计值)
        权重 = multiply(权重, exp(指数迭代量))
        权重 /= 权重.sum()
        累计估计值 += 结果权重 * 类别估计值
        #print('迭代%d累计估计值:'%i, 累计估计值)
        累计错误值 = multiply(sign(累计估计值) != mat(标签).T, ones((个数, 1)))
        错误率 = 累计错误值.sum() / 个数
        print('迭代%d错误率:'%i, 错误率)
        if 错误率 <= 0.0:
            break
    return 弱分类器们, 累计估计值

def ada分类(待分类数据, 弱分类器们):
    from numpy import mat, shape, zeros, sign
    特征矩阵 = mat(待分类数据)
    个数 = shape(特征矩阵)[0]
    累计类别估计 = mat(zeros((个数, 1)))
    for i in range(len(弱分类器们)):
        类别估计 = 决策树分类(特征矩阵, 弱分类器们[i]['划分维度'], 弱分类器们[i]['阈值'], 弱分类器们[i]['判定范围'])
        累计类别估计 += 弱分类器们[i]['结果权重'] * 类别估计
        #print(累计类别估计)
    return sign(累计类别估计)

def 加载数据集(文件路径):
    特征 = []
    标签 = []
    文件 = open(文件路径)
    for 行 in 文件.readlines():
        特征.append([float(元素) for 元素 in 行.strip().split('\t')[:-1]])
        标签.append(float(行.strip().split('\t')[-1]))
    文件.close()
    return 特征, 标签

def 绘制二分类器工作特性曲线(预测强度, 标签):
    from numpy import array
    from matplotlib import pyplot
    pyplot.rcParams['font.sans-serif']=['SimHei']#设置中文字体
    界面 = pyplot.figure()
    界面.clf()
    图像 = pyplot.subplot(111)
    光标 = (1.0, 1.0)
    强度索引 = 预测强度.argsort()
    x方向步长 = 1 / sum(array(标签)!=1.0)
    y方向步长 = 1 / sum(array(标签)==1.0)
    曲线下面积 = 0
    for 索引 in 强度索引.tolist()[0]:
        if 标签[索引] == 1.0:
            x变化 = 0
            y变化 = y方向步长
        else:
            x变化 = x方向步长
            y变化 = 0
            曲线下面积 += 光标[1]
        图像.plot([光标[0], 光标[0]-x变化], [光标[1], 光标[1]-y变化], c='b')
        光标 = (光标[0]-x变化, 光标[1]-y变化)
    pyplot.xlabel(u'假阳率'); pyplot.ylabel(u'真阳率')
    pyplot.title('ROC curve for AdaBoost horse colic detection system')
    图像.axis([0,1,0,1])
    pyplot.show()
    print ("曲线下的面积为:", 曲线下面积*x方向步长)

特征矩阵, 类别 = 载入小样例()
弱分类器们, 累计估计值 = 训练ada提升(特征矩阵, 类别)
print(ada分类([[0,0],[5,5]], 弱分类器们))

训练特征, 训练标签 = 加载数据集('../data/ch07_horseColicTraining2.txt')
弱分类器们, 累计估计值 = 训练ada提升(训练特征, 训练标签)
测试特征, 测试标签 = 加载数据集('../data/ch07_horseColictest2.txt')
预测标签 = ada分类(测试特征, 弱分类器们)
错误标签 = 预测标签 != numpy.mat(测试标签).T
print('horseColic错误率:', 错误标签.sum() / len(错误标签))
绘制二分类器工作特性曲线(累计估计值.T, 训练标签)