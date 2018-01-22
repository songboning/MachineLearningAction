# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 17:09:38 2018
自编码器
@author: Burning
"""

""" *** 初始化 *** """
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from functools import partial#函数修饰器,可以固定一部分参数从而减少代码量
from tensorflow.examples.tutorials.mnist import input_data

def reset_env(seed = 42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)
#用来让某部分的随机化效果和参考书一致,每个部分都应当用这个函数先初始化

def 绘图(图像, 形状=[28,28]):
    plt.imshow(图像.reshape(形状), cmap='Greys', interpolation='nearest')
    plt.axis('off')

he_init = tf.contrib.layers.variance_scaling_initializer()#He参数初始化
mnist = input_data.read_data_sets("../data/mnist")

""" *** 多层自编码器 *** """
reset_env()
n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 150
n_hidden3 = n_hidden1
n_outputs = n_inputs
学习率 = 0.01
L2正则化范围 = 0.0001

X = tf.placeholder(tf.float32, shape=[None, n_inputs], name='X')
L2正则化器 = tf.contrib.layers.l2_regularizer(L2正则化范围)
稠密层定义器 = partial(tf.layers.dense, activation=tf.nn.elu, kernel_initializer=he_init, kernel_regularizer=L2正则化器)#简化后续稠密(全连接)层定义代码
hidden1 = 稠密层定义器(X, n_hidden1)
hidden2 = 稠密层定义器(hidden1, n_hidden2)#code
hidden3 = 稠密层定义器(hidden2, n_hidden3)
outputs = 稠密层定义器(hidden3, n_outputs)
重构损失 = tf.reduce_mean(tf.square(outputs - X))
正则损失 = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
损失 = tf.add_n([重构损失] + 正则损失, name='loss')
优化器 = tf.train.AdamOptimizer(学习率)
训练操作 = 优化器.minimize(损失)
init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epochs = 5
batch_size = 150
with tf.Session() as session:
    init.run()
    for epoch in range(n_epochs):
        n_batches = mnist.train.num_examples // batch_size
        for i in range(n_batches):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            session.run(训练操作, feed_dict={X: X_batch})
        损失值 = 重构损失.eval(feed_dict={X: X_batch})
        print(epoch, '均方误差:', 损失值)
        saver.save(session, '../data/II15/mnist_model')

def 展示重构数字(X, outputs, model_path='../data/II15/mnist_model', n_test_digits=2):
    with tf.Session() as session:
        if model_path:
            saver.restore(session, model_path)
        X_test = mnist.test.images[:n_test_digits]
        outputs_val = outputs.eval(feed_dict={X: X_test})

    fig = plt.figure(figsize=(8, 3 * n_test_digits))
    for digit_index in range(n_test_digits):
        plt.subplot(n_test_digits, 2, digit_index * 2 + 1)
        绘图(X_test[digit_index])
        plt.subplot(n_test_digits, 2, digit_index * 2 + 2)
        绘图(outputs_val[digit_index])
#将自编码器学得的编码表示展现出来

展示重构数字(X, outputs)