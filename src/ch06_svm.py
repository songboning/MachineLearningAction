# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 22:15:46 2017
核方法的支持向量机模板
@author: Burning
"""

import numpy
from numpy import mat, shape

def 加载数据集(文件路径):
    特征 = []
    标签 = []
    文件 = open(文件路径)
    for 行 in 文件.readlines():
        特征.append([float(元素) for 元素 in 行.strip().split('\t')[:-1]])
        标签.append(float(行.strip().split('\t')[-1]))
    文件.close()
    return mat(特征), mat(标签).T

def 核变换(X, A, 核信息元组):
    from numpy import zeros, exp
    行数 = shape(X)[0]
    核 = mat(zeros((行数, 1)))
    if 核信息元组[0] == '线性':
        核 = X * A.T
    elif 核信息元组[0] == '径向基':
        for i in range(行数):
            差分行 = X[i,:] - A
            核[i] = 差分行 * 差分行.T
        核 = exp(核 / (-1 * 核信息元组[1] ** 2))
    else:
        raise NameError('没有预备的核方法:%s' %核信息元组[0])
    return 核

class 优化方法内数据结构:
    def __init__(self, 特征矩阵, 标签矩阵, 正则化强度, 容错率, 核信息元组):
        from numpy import zeros
        self.X = 特征矩阵
        self.Y = 标签矩阵
        self.C = 正则化强度
        self.tor = 容错率
        self.m = shape(特征矩阵)[0]
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        self.Ecache = mat(zeros((self.m, 2)))
        self.K = mat(zeros((self.m,self.m)))
        for i in range(self.m):
            self.K[:,i] = 核变换(self.X, self.X[i,:], 核信息元组)

def 选择随机那个(这个, 总个数):
    from numpy.random import randint
    那个 = 这个
    while(那个 == 这个):
        那个 = randint(总个数)
    return 那个

def 选择最优那个(这个, 数据, Ei):
    from numpy import nonzero
    最优那个 = -1
    最大E差 = 0
    Ej = 0
    数据.Ecache[这个] = [1,Ei]
    可用Ecache列表 = nonzero(数据.Ecache[:,0].A)[0]
    if len(可用Ecache列表) > 1:
        for 备选那个 in 可用Ecache列表:
            if 备选那个 == 这个:
                continue
            Ek = 计算Ek(数据, 备选那个)
            deltaE = abs(Ei - Ek)
            if (deltaE > 最大E差):
                最优那个 = 备选那个; 最大E差 = deltaE; Ej = Ek
        return 最优那个, Ej
    else:
        随机那个 = 选择随机那个(这个, 数据.m)
        Ej = 计算Ek(数据, 随机那个)
        return 随机那个, Ej

def 约束alpha(a, 下界, 上界):
    if a > 上界:
        a = 上界
    if a < 下界:
        a = 下界
    return a

def 计算Ek(数据, k):
    from numpy import multiply
    fXk = float(multiply(数据.alphas, 数据.Y).T * 数据.K[:,k] + 数据.b)
    Ek = fXk - float(数据.Y[k])
    return Ek

def 更新Ek(数据, k):
    Ek = 计算Ek(数据, k)
    数据.Ecache[k] = [1, Ek]

def 内循环(这个, 数据):
    Ei = 计算Ek(数据, 这个)
    if 数据.Y[这个] * Ei < -数据.tor and 数据.alphas[这个] < 数据.C or 数据.Y[这个] * Ei > 数据.tor and 数据.alphas[这个] > 0:
        那个, Ej = 选择最优那个(这个, 数据, Ei)
        alphaIold = 数据.alphas[这个].copy()
        alphaJold = 数据.alphas[那个].copy()
        if 数据.Y[这个] != 数据.Y[那个]:
            下界 = max(0, 数据.alphas[那个] - 数据.alphas[这个])
            上界 = min(数据.C, 数据.C + 数据.alphas[那个] - 数据.alphas[这个])
        else:
            下界 = max(0, 数据.alphas[那个] + 数据.alphas[这个] - 数据.C)
            上界 = min(数据.C, 数据.alphas[那个] + 数据.alphas[这个])
        if 下界 == 上界:
            print('下界==上界')
            return 0
        eta = 2 * 数据.K[这个, 那个] - 数据.K[这个, 这个] - 数据.K[那个, 那个]
        if eta >= 0:
            print('eta>=0')
            return 0
        数据.alphas[那个] -= 数据.Y[那个] * (Ei - Ej) / eta
        数据.alphas[那个] = 约束alpha(数据.alphas[那个], 下界, 上界)
        更新Ek(数据, 那个)
        if abs(数据.alphas[那个] - alphaJold) < 1e-5:
            print('j not move enough')
            return 0
        数据.alphas[这个] += 数据.Y[那个] * 数据.Y[这个] * (alphaJold - 数据.alphas[那个])
        更新Ek(数据, 这个)
        b1 = 数据.b - Ei - 数据.Y[这个] * (数据.alphas[这个] - alphaIold) * 数据.K[这个,这个] - 数据.Y[那个] * (数据.alphas[那个] - alphaJold) * 数据.K[这个,那个]
        b2 = 数据.b - Ej - 数据.Y[这个] * (数据.alphas[这个] - alphaIold) * 数据.K[这个,那个] - 数据.Y[那个] * (数据.alphas[那个] - alphaJold) * 数据.K[那个,那个]
        if 0 < 数据.alphas[这个] and 数据.alphas[这个] < 数据.C:
            数据.b = b1
        elif 0 < 数据.alphas[那个] and 数据.alphas[那个] < 数据.C:
            数据.b = b2
        else:
            数据.b = (b1 + b2) / 2
        return 1
    else:
        return 0

def 序列最小优化(特征矩阵, 标签矩阵, 正则化强度, 容错率, 最大迭代次数, 核信息元组=('线性', 0)):
    from numpy import nonzero
    数据 = 优化方法内数据结构(特征矩阵, 标签矩阵, 正则化强度, 容错率, 核信息元组)
    迭代次数 = 0
    是否全集 = True
    alpha对被改变 = 0
    while 迭代次数 < 最大迭代次数 and (alpha对被改变 > 0 or 是否全集):
        alpha对被改变 = 0
        if 是否全集:
            for i in range(数据.m):
                alpha对被改变 += 内循环(i, 数据)
                print('全集, 迭代次数:%d\ti:%d\t改变对:%d' %(迭代次数, i, alpha对被改变))
            迭代次数 += 1
        else:
            非边界 = nonzero((数据.alphas.A > 0) * (数据.alphas.A < 数据.C))[0]
            for i in 非边界:
                alpha对被改变 += 内循环(i, 数据)
                print('non-bound, 迭代次数:%d\ti:%d\t改变对:%d' %(迭代次数, i, alpha对被改变))
            迭代次数 += 1
        if 是否全集:
            是否全集 = False
        elif not alpha对被改变:
            是否全集 = True
        print('迭代次数:%d' %迭代次数)
    return 数据.b, 数据.alphas

def 测试径向基核(k1=1.3):
    from numpy import nonzero, multiply, sign
    特征矩阵, 标签矩阵 = 加载数据集('../data/ch06_testSetRBF.txt')
    b, alphas = 序列最小优化(特征矩阵, 标签矩阵, 200, 0.0001, 10000, ('径向基', k1))
    支持向量索引 = nonzero(alphas.A>0)[0]
    支持向量 = 特征矩阵[支持向量索引]
    支持标签 = 标签矩阵[支持向量索引]
    print("有%d个支持向量" %shape(支持向量)[0])
    m, n = shape(特征矩阵)
    错误数 = 0
    for i in range(m):
        kernelEval = 核变换(支持向量, 特征矩阵[i,:], ('径向基', k1))
        预测 = kernelEval.T * multiply(支持标签, alphas[支持向量索引]) + b
        if sign(预测) != sign(标签矩阵[i]):
            错误数 += 1
    print("训练错误率: %f" %(float(错误数)/m))
    特征矩阵, 标签矩阵 = 加载数据集('../data/ch06_testSetRBF2.txt')
    错误数 = 0
    m,n = shape(特征矩阵)
    for i in range(m):
        kernelEval = 核变换(支持向量, 特征矩阵[i,:], ('径向基', k1))
        预测=kernelEval.T * multiply(支持标签, alphas[支持向量索引]) + b
        if sign(预测)!=sign(标签矩阵[i]):
            错误数 += 1
    print("测试错误率: %f" %(float(错误数)/m))   