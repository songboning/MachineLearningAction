#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 20:28:11 2017
朴素贝叶斯模版
@author: songboning
"""

import numpy
import re
import random
import operator
import feedparser

def 小样例_留言():
    留言=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    标签_是否侮辱 = [0,1,0,1,0,1]
    return 留言, 标签_是否侮辱

def 构造词表(数据集):
    词表 = set()
    for 文本 in 数据集:
        词表 |= set(文本)
    return list(词表)

def 由文本构造词向量(词表, 文本):
    词向量 = [0] * len(词表)
    for 词 in 文本:
        if 词 in 词表:
            词向量[词表.index(词)] = 1
        else:
            print('警告: %s 不在词表中' %词)
    return 词向量

def 由文本构造词袋向量(词表, 文本):
    词向量 = [0] * len(词表)
    for 词 in 文本:
        if 词 in 词表:
            词向量[词表.index(词)] += 1
        #else:
        #    print('警告: %s 不在词表中' %词)
    return 词向量

def 训练朴素贝叶斯二分类器(训练矩阵, 训练类别):
    from numpy import zeros, ones, log
    文档数 = len(训练矩阵)
    词数 = len(训练矩阵[0])
    总为真概率 = sum(训练类别) / 文档数
    词使得类别假个数 = ones(词数)#zeros(词数)
    词使得类别真个数 = ones(词数)#zeros(词数)
    使得类别假总词数 = 2
    使得类别真总词数 = 2#这个分母概率的计算并不严格符合高斯假设
    for i, 词向量 in enumerate(训练矩阵):
        if 训练类别[i]:
            词使得类别真个数 += 词向量
            使得类别真总词数 += sum(词向量)
        else:
            词使得类别假个数 += 词向量
            使得类别假总词数 += sum(词向量)
    词置真概率 = log(词使得类别真个数 / 使得类别真总词数)#词使得类别真个数 / 使得类别真总词数
    词置假概率 = log(词使得类别假个数 / 使得类别假总词数)#词使得类别假个数 / 使得类别假总词数
    return 词置假概率, 词置真概率, 总为真概率

def 朴素贝叶斯二分类(待分类词向量, 词置假概率, 词置真概率, 总为真概率):
    from numpy import log
    为真概率 = sum(待分类词向量 * 词置真概率) + log(总为真概率)
    为假概率 = sum(待分类词向量 * 词置假概率) + log(1.0 - 总为真概率)
    return 为真概率 > 为假概率

def 测试朴素贝叶斯二分类器():
    留言集, 文风 = 小样例_留言()
    词表 = 构造词表(留言集)
    训练矩阵 = []
    for 留言 in 留言集:
        训练矩阵.append(由文本构造词向量(词表, 留言))
    词置假概率, 词置真概率, 总为真概率 = 训练朴素贝叶斯二分类器(训练矩阵, 文风)
    测试文本 = ['love', 'my', 'dalmation']
    词向量 = 由文本构造词向量(词表, 测试文本)
    print('%s 是'%测试文本, 朴素贝叶斯二分类(词向量, 词置假概率, 词置真概率, 总为真概率))
    测试文本 = ['stupid', 'garbage']
    词向量 = 由文本构造词向量(词表, 测试文本)
    print('%s 是'%测试文本, 朴素贝叶斯二分类(词向量, 词置假概率, 词置真概率, 总为真概率))

def 文本解析(文本串):
    符号集 = re.split(r'\W*', 文本串)#大写W表示非字母数字_
    return [符号.lower() for 符号 in 符号集 if len(符号) > 2]

def 测试垃圾邮件分类():
    邮件集 = []
    是否垃圾 = []
    for i in range(1,26):
        邮件 = 文本解析(open('../data/ch04_email/spam/%d.txt' %i, 'rb').read().decode('Windows-1252'))
        邮件集.append(邮件)
        是否垃圾.append(1)
        邮件 = 文本解析(open('../data/ch04_email/ham/%d.txt' %i, 'rb').read().decode('Windows-1252'))
        邮件集.append(邮件)
        是否垃圾.append(0)
    词表 = 构造词表(邮件集)
    训练集索引 = list(range(50))
    测试集索引=[]
    for i in range(10):
        随机编号 = int(random.uniform(0, len(训练集索引)))
        测试集索引.append(训练集索引[随机编号])
        del(训练集索引[随机编号])  
    训练矩阵=[]; 训练标签 = []
    for 数据索引 in 训练集索引:
        训练矩阵.append(由文本构造词袋向量(词表, 邮件集[数据索引]))
        训练标签.append(是否垃圾[数据索引])
    词置假概率, 词置真概率, 总为真概率 = 训练朴素贝叶斯二分类器(训练矩阵, 训练标签)
    错误数 = 0
    for 数据索引 in 测试集索引:
        词向量 = 由文本构造词袋向量(词表, 邮件集[数据索引])
        if 朴素贝叶斯二分类(词向量, 词置假概率, 词置真概率, 总为真概率) != 是否垃圾[数据索引]:
            错误数 += 1
            #print ("分类错误:", 邮件集[数据索引])
    错误率 = float(错误数)/len(测试集索引)
    print ('垃圾邮件分类错误率: ', 错误率)
    return 错误率

def 计算相对高频词(词表, 全文):
    from operator import itemgetter
    频率字典 = {}
    for 符号 in 词表:
        频率字典[符号] = 全文.count(符号)
    return sorted(频率字典.items(), key=itemgetter(1), reverse=True)[:30]

def 测试地域分类(数据流0, 数据流1):
    数据流公共长度 = min(len(数据流0['entries']), len(数据流1['entries']))#用于类别平衡
    文档列表 = []
    标签 = []
    文本集 = []
    for i in range(数据流公共长度):
        抽样文本 = 文本解析(数据流0['entries'][i]['summary'])
        文档列表.append(抽样文本)
        标签.append(0)
        文本集.extend(抽样文本)
        抽样文本 = 文本解析(数据流1['entries'][i]['summary'])
        文档列表.append(抽样文本)
        标签.append(1)
        文本集.extend(抽样文本)
    词表 = 构造词表(文档列表)
    最热30词频 = 计算相对高频词(词表, 文本集)
    for 词频 in 最热30词频:
        if 词频[0] in 词表:
            词表.remove(词频[0])
    训练集索引 = list(range(2*数据流公共长度))
    测试集索引 = []
    for i in range(int(数据流公共长度*0.6)):
        随机编号 = random.randint(0, len(训练集索引)-1)
        测试集索引.append(训练集索引[随机编号])
        del(训练集索引[随机编号])
    训练矩阵 = []
    训练标签 = []
    for 文档索引 in 训练集索引:
        训练矩阵.append(由文本构造词袋向量(词表, 文档列表[文档索引]))
        训练标签.append(标签[文档索引])
    置0概率, 置1概率, 总体为1概率 = 训练朴素贝叶斯二分类器(训练矩阵, 训练标签)
    错误统计 = 0
    for 文档索引 in 测试集索引:
        词袋向量 = 由文本构造词袋向量(词表, 文档列表[文档索引])
        if 朴素贝叶斯二分类(词袋向量, 置0概率, 置1概率, 总体为1概率) != 标签[文档索引]:
            错误统计 += 1
    print('地域分类错误率:', 错误统计/len(测试集索引))
    return 词表, 置0概率, 置1概率

def 得到特征词(数据流0, 数据流1):
    词表, 置0概率, 置1概率 = 测试地域分类(数据流0, 数据流1)
    数据流0特征词 = []
    数据流1特征词 = []
    for i in range(len(置0概率)):
        #if 置0概率[i] > -0.6:
            数据流0特征词.append((词表[i], 置0概率[i]))
        #if 置1概率[i] > -0.6:
            数据流1特征词.append((词表[i], 置1概率[i]))
    有序的数据流0特征词 = sorted(数据流0特征词, key=lambda pair:pair[1], reverse=True)
    有序的数据流1特征词 = sorted(数据流1特征词, key=lambda pair:pair[1], reverse=True)
    return 有序的数据流0特征词[:10], 有序的数据流1特征词[:10]

测试朴素贝叶斯二分类器()
测试垃圾邮件分类()
ny=feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
sf=feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
得到特征词(ny,sf)