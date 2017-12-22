# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 17:42:35 2017
回归树与模型树模板
@author: Burning
"""

import numpy
from numpy import mat

def 加载数据集(文件路径):
    数据集 = []
    文件 = open(文件路径)
    for 行 in 文件.readlines():
        数据集.append(list(map(float, 行.strip().split('\t'))))
    文件.close()
    return mat(数据集)
#默认最后一列标签

def 二分数据矩阵(数据矩阵, 特征维度, 划分值):
    from numpy import nonzero
    大的 = 数据矩阵[nonzero(数据矩阵[:,特征维度] > 划分值)[0], :]
    小的 = 数据矩阵[nonzero(数据矩阵[:,特征维度] <= 划分值)[0], :]
    return 大的, 小的

def 均值叶节点(数据矩阵):
    return numpy.mean(数据矩阵[:,-1])

def 平方误差(数据矩阵):
    return numpy.var(数据矩阵[:,-1]) * numpy.shape(数据矩阵)[0]

def 选择最佳划分(数据矩阵, 叶节点函数=均值叶节点, 损失函数=平方误差, 最少误差下降值=1, 最少切分样本数=4):
    if len(set(数据矩阵[:,-1].T.tolist()[0])) == 1:
        return None, 叶节点函数(数据矩阵)
    from numpy import shape, inf
    特征数 = shape(数据矩阵)[1] - 1
    当前损失 = 损失函数(数据矩阵)
    最小损失 = inf
    for 维度 in range(特征数):
        for 划分值 in set(数据矩阵[:,维度].T.tolist()[0]):
            大的, 小的 = 二分数据矩阵(数据矩阵, 维度, 划分值)
            if shape(大的)[0] < 最少切分样本数 or shape(小的)[0] < 最少切分样本数:
                continue
            新的损失 = 损失函数(大的) + 损失函数(小的)
            if 新的损失 < 最小损失:
                最小损失 = 新的损失
                最佳划分维度 = 维度
                最佳划分值 = 划分值
    if (当前损失 - 最小损失) < 最少误差下降值:
        return None, 叶节点函数(数据矩阵)
    return 最佳划分维度, 最佳划分值

def 构造回归树(数据矩阵, 叶节点函数=均值叶节点, 损失函数=平方误差, 最少误差下降值=1, 最少切分样本数=4):
    划分维度, 划分值 = 选择最佳划分(数据矩阵, 叶节点函数, 损失函数, 最少误差下降值, 最少切分样本数)
    if 划分维度 == None:#这里不能写成if not形式,因为划分维度会为0
        return 划分值
    节点 = dict()
    节点['划分维度'] = 划分维度
    节点['划分值'] = 划分值
    大的, 小的 = 二分数据矩阵(数据矩阵, 划分维度, 划分值)
    节点['大儿子'] = 构造回归树(大的, 叶节点函数, 损失函数, 最少误差下降值, 最少切分样本数)
    节点['小儿子'] = 构造回归树(小的, 叶节点函数, 损失函数, 最少误差下降值, 最少切分样本数)
    return 节点

def 均值化回归树(回归树):
    if type(回归树['大儿子']) == dict:
        回归树['大儿子'] = 均值化回归树(回归树['大儿子'])
    if type(回归树['小儿子']) == dict:
        回归树['小儿子'] = 均值化回归树(回归树['小儿子'])
    return (回归树['大儿子'] + 回归树['小儿子']) / 2

def 后剪枝(回归树, 测试集):
    from numpy import shape, sum, power
    if not shape(测试集)[0]:
        return 均值化回归树(回归树)
    if type(回归树['大儿子']) == dict or type(回归树['小儿子']) == dict:
        大的, 小的 = 二分数据矩阵(测试集, 回归树['划分维度'], 回归树['划分值'])
    if type(回归树['大儿子']) == dict:
        回归树['大儿子'] = 后剪枝(回归树['大儿子'], 大的)
    if type(回归树['小儿子']) == dict:
        回归树['小儿子'] = 后剪枝(回归树['小儿子'], 小的)
    if type(回归树['大儿子']) != dict and type(回归树['小儿子']) != dict:
        大的, 小的 = 二分数据矩阵(测试集, 回归树['划分维度'], 回归树['划分值'])
        不合并的损失 = sum(power(大的[:,-1] - 回归树['大儿子'], 2)) + sum(power(小的[:,-1] - 回归树['小儿子'], 2))
        均值 = (回归树['大儿子'] + 回归树['小儿子']) / 2
        合并的损失 = sum(power(测试集[:,-1], 2))
        if 合并的损失 < 不合并的损失:
            print ('合并')
            return 均值
    return 回归树

def 线性回归(数据矩阵):
    from numpy import shape, linalg, ones
    特征维数 = shape(数据矩阵)[1]
    x = mat(ones(shape(数据矩阵)))
    x[:,1:特征维数] = 数据矩阵[:,:特征维数-1]
    #数据固定第一维恒为1可用于统一计算y=wx+b中的常数偏移b
    #x = 数据矩阵[:,:特征维数-1]
    y = 数据矩阵[:,-1]
    xTx = x.T * x
    if not linalg.det(xTx):
        raise ValueError('矩阵奇异,不可逆,请增大最少切分样本数')
    回归系数 = xTx.I * (x.T * y)
    return 回归系数, x, y

def 线性叶节点(数据矩阵):
    return 线性回归(数据矩阵)[0]

def 回归误差(数据矩阵):
    from numpy import sum, power
    回归系数, 特征, 标签 = 线性回归(数据矩阵)
    预测 = 特征 * 回归系数
    return sum(power(预测 - 标签, 2))

def 均值叶估计(叶节点模型, 特征):
    return float(叶节点模型)

def 线性叶估计(叶节点模型, 特征):
    from numpy import shape, ones
    特征维数 = shape(特征)[1]
    x = mat(ones((1, 特征维数+1)))
    x[:, 1:特征维数+1] = 特征
    return float(x * 叶节点模型)

def 回归树预测(回归树, 特征, 叶估计模型=均值叶估计):
    if type(回归树) != dict:
        return 叶估计模型(回归树, 特征)
    if 特征[回归树['划分维度']] > 回归树['划分值']:
        return 回归树预测(回归树['大儿子'], 特征, 叶估计模型)
    else:
        return 回归树预测(回归树['小儿子'], 特征, 叶估计模型)

def 回归树测试(回归树, 测试数据, 叶估计模型=均值叶估计):
    from numpy import shape, zeros
    个数 = shape(测试数据)[0]
    预测值 = mat(zeros((个数, 1)))
    for i in range(个数):
        预测值[i, 0] = 回归树预测(回归树, 测试数据[i], 叶估计模型)
    return 预测值

数据矩阵 = 加载数据集('../data/ch09_ex2.txt')
回归树 = 构造回归树(数据矩阵)
print ('回归树:', 回归树)
测试矩阵 = 加载数据集('../data/ch09_ex2test.txt')
回归树 = 后剪枝(回归树, 测试矩阵)
print ('剪枝后的回归树:', 回归树)

train = 加载数据集('../data/ch09_bikeSpeedVsIq_train.txt')
test = 加载数据集('../data/ch09_bikeSpeedVsIq_test.txt')
回归树 = 构造回归树(train, 最少切分样本数=20)
预测值 = 回归树测试(回归树, test[:,:-1], 均值叶估计)
相关系数 = numpy.corrcoef(预测值, test[:,-1], rowvar=0)#R^2检验
print ('回归树效果', 相关系数[0,1])
模型树 = 构造回归树(train, 线性叶节点, 回归误差, 最少切分样本数=20)
预测值 = 回归树测试(模型树, test[:,:-1], 线性叶估计)
相关系数 = numpy.corrcoef(预测值, test[:,-1], rowvar=0)
print ('模型树效果', 相关系数[0,1])