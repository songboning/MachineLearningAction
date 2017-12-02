#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 14:25:18 2017
决策树模版及相关代码,数据集默认最后一列是标签
@author: songboning
"""

import math
import operator
import matplotlib

def 构造小样例():
    数据集 = [[1,1,'yes'],
           [1,1,'yes'],
           [1,0,'no'],
           [0,1,'no'],
           [0,1,'no']]
    特征名 = ['no surfacing', 'flippers']
    return 数据集, 特征名

def 样例决策树(i):
    listOfTrees =[{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
              {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
              ]
    return listOfTrees[i]

def 计算香农熵(数据集):
    from math import log
    实体数目 = len(数据集)
    标签计数 = {}
    for 数据项 in 数据集:
        标签 = 数据项[-1]
        标签计数[标签] = 标签计数.get(标签, 0) + 1
    熵 = 0.0
    for 标签 in 标签计数:
        标签的概率 = 标签计数[标签] / 实体数目
        熵 -= 标签的概率 * log(标签的概率, 2)
    return 熵

def 按特征提取子集(待划分数据集, 划分特征列, 划分特征值):
    目标子集 = []
    for 数据项 in 待划分数据集:
        if 数据项[划分特征列] == 划分特征值:
            子项 = 数据项[:划分特征列]#这里不能省略,否则第0列划分会有问题
            子项.extend(数据项[划分特征列+1:])
            目标子集.append(子项)
    return 目标子集

def 选择最佳划分特征(数据集):
    特征数 = len(数据集[0]) - 1
    基本熵 = 计算香农熵(数据集)
    最佳信息增益 = 0.0
    最佳特征 = -1
    for i in range(特征数):
        特征取值 = [数据项[i] for 数据项 in 数据集]
        特征值散列 = set(特征取值)
        新熵 = 0.0
        for 值 in 特征值散列:
            数据子集 = 按特征提取子集(数据集, i, 值)
            特征值概率 = len(数据子集) / len(数据集)
            新熵 += 特征值概率 * 计算香农熵(数据子集)
        信息增益 = 基本熵 - 新熵
        if(信息增益 > 最佳信息增益):
            最佳信息增益 = 信息增益
            最佳特征 = i
    return 最佳特征

def 最大频次(标签列):
    from operator import itemgetter
    标签计数 = {}
    for 标签 in 标签列:
        标签计数[标签] = 标签计数.get(标签, 0) + 1
    有序标签计数 = sorted(标签计数.items(), key=itemgetter(1), reverse=True)
    return 有序标签计数[0][0]

def 构造决策树(数据集, 特征名):
    类别 = [数据项[-1] for 数据项 in 数据集]
    if 类别.count(类别[0]) == len(类别):
        return 类别[0]
    if len(数据集[0]) == 1:
        return 最大频次(类别)
    最佳划分特征列 = 选择最佳划分特征(数据集)
    最佳划分特征名 = 特征名[最佳划分特征列]
    决策树 = {最佳划分特征名:{}}
    #del(特征名[最佳划分特征列])
    划分特征值 = set([数据项[最佳划分特征列] for 数据项 in 数据集])
    for 值 in 划分特征值:
        剩余特征 = 特征名[:最佳划分特征列]
        剩余特征.extend(特征名[最佳划分特征列+1:])
        数据子集 = 按特征提取子集(数据集, 最佳划分特征列, 值)
        决策树[最佳划分特征名][值] = 构造决策树(数据子集, 剩余特征)
    return 决策树

def 决策树分类(决策树, 特征名称, 特征向量):
    划分特征名 = list(决策树.keys())[0]
    子树们 = 决策树[划分特征名]
    划分特征索引 = 特征名称.index(划分特征名)
    for 特征值 in 子树们.keys():
        if 特征向量[划分特征索引] == 特征值:
            if type(子树们[特征值]) == dict:
                return 决策树分类(子树们[特征值], 特征名称, 特征向量)
            else:
                return 子树们[特征值]

def 得到叶子个数(决策树):
    叶子个数 = 0
    #划分特征 = 决策树.keys()[0]
    子树结构 = list(决策树.values())[0]
    for 子树 in 子树结构.values():
        if type(子树) == dict:
            叶子个数 += 得到叶子个数(子树)
        else:
            叶子个数 += 1
    return 叶子个数

def 得到树深度(决策树):
    最大深度 = 0
    #划分特征 = 决策树.keys()[0]
    子树结构 = list(决策树.values())[0]
    for 子树 in 子树结构.values():
        if type(子树) == dict:
            当前深度 = 1 + 得到树深度(子树)
        else:
            当前深度 = 1
        最大深度 = max(最大深度, 当前深度)
    return 最大深度

def 画节点(nodeTxt, centerPt, parentPt, 节点类型):
    决策树画板.图像.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',
             xytext=centerPt, textcoords='axes fraction',
             va="center", ha="center", bbox=节点类型, arrowprops=箭头形式)

def 填充枝文本(中心点, 父节点, 文本):
    xMid = (父节点[0]-中心点[0])/2.0 + 中心点[0]
    yMid = (父节点[1]-中心点[1])/2.0 + 中心点[1]
    决策树画板.图像.text(xMid, yMid, 文本, va="center", ha="center", rotation=30)

def 绘制树(决策树, 父节点位置, 划分值):
    宽度 = 得到叶子个数(决策树)
    #深度 = 得到树深度(决策树)
    划分特征名 = list(决策树.keys())[0]
    中心点 = (绘制树.x基 + (1.0 + 宽度)/2/绘制树.总宽度, 绘制树.y基)
    填充枝文本(中心点, 父节点位置, 划分值)
    画节点(划分特征名, 中心点, 父节点位置, 决策点)
    子树 = 决策树[划分特征名]
    绘制树.y基 -= 1/绘制树.总深度
    for 特征值 in 子树.keys():
        if type(子树[特征值]) == dict:
            绘制树(子树[特征值], 中心点, str(特征值))
        else:
            绘制树.x基 += 1/绘制树.总宽度
            画节点(子树[特征值], (绘制树.x基, 绘制树.y基), 中心点, 叶节点)
            填充枝文本((绘制树.x基, 绘制树.y基), 中心点, str(特征值))
    绘制树.y基 += 1/绘制树.总深度

def 决策树画板(决策树):
    from matplotlib import pyplot as plt
    窗口 = plt.figure(1, facecolor='white')
    窗口.clf()
    axprops = dict(xticks=[], yticks=[])
    #不要边框和坐标轴
    决策树画板.图像 = plt.subplot(111, frameon=False, **axprops)
    绘制树.总宽度 = 得到叶子个数(决策树)
    绘制树.总深度 = 得到树深度(决策树)
    绘制树.x基 = -0.5/绘制树.总宽度
    绘制树.y基 = 1.0
    绘制树(决策树, (0.5,1.0), '')
    决策点 = dict(boxstyle="sawtooth", fc="0.8")
    叶节点 = dict(boxstyle="round4", fc="0.8")
    arrow_args = dict(arrowstyle="<-")
    plt.show()

决策点 = dict(boxstyle="sawtooth", fc="0.8")
叶节点 = dict(boxstyle="round4", fc="0.8")
箭头形式 = dict(arrowstyle="<-")

try:
    数据文件 = open('../data/ch03_lenses.txt')
    数据集 = [inst.strip().split('\t') for inst in 数据文件.readlines()]
    数据文件.close()
    特征名 = ['age', 'prescript', 'astigmatic', 'tearrate']
except Exception:
    print(Exception)
    数据集, 特征名 = 构造小样例()
决策树 = 构造决策树(数据集, 特征名)
决策树画板(决策树)
print(决策树分类(决策树, 特征名, 数据集[0]))