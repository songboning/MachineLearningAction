# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 14:13:21 2018
人工神经网络基础
@author: Burning
"""

""" *** 初始化 *** """
import os
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

def reset_env(seed = 42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)
#用来让某部分的随机化效果和参考书一致,每个部分都应当用这个函数先初始化

mnist = input_data.read_data_sets('../data/mnist')

""" *** 朴素神经网络 *** """
def 神经元层(X, 神经元个数, name, 激活函数=None):
    with tf.name_scope(name):#使用命名空间为了美观
        输入数 = int(X.get_shape()[1])
        标准差 = 2 / np.sqrt(输入数)#使用这个标准差初始化期望会让后续计算快一些
        initW = tf.truncated_normal((输入数, 神经元个数), stddev=标准差)
        W = tf.Variable(initW, name='kernel')
        b = tf.Variable(tf.zeros([神经元个数]), name='bias')
        Z = tf.matmul(X, W) + b
        if 激活函数:
            return 激活函数(Z)
        return Z
#tensorflow神经元层的朴素实现

输入数 = 28 * 28
输出数 = 10
学习率 = 0.01
X = tf.placeholder(tf.float32, shape=(None, 输入数), name='X')
y = tf.placeholder(tf.int64, shape=(None), name='y')
with tf.name_scope('dnn'):
    隐藏层1 = 神经元层(X, 300, name='hidden1', 激活函数=tf.nn.relu)
    隐藏层2 = 神经元层(隐藏层1, 100, name='hidden2', 激活函数=tf.nn.relu)
    输出层 = 神经元层(隐藏层2, 输出数, name='outputs')#未归一化概率
with tf.name_scope('loss'):
    交叉熵 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=输出层)
    损失 = tf.reduce_mean(交叉熵, name='loss')
with tf.name_scope('train'):
    优化器 = tf.train.GradientDescentOptimizer(学习率)
    训练 = 优化器.minimize(损失)
with tf.name_scope('eval'):
    正确否 = tf.nn.in_top_k(输出层, y, 1)
    准确率 = tf.reduce_mean(tf.cast(正确否, tf.float32))

init = tf.global_variables_initializer()
存储器 = tf.train.Saver()
训练次数 = 10
批量 = 50
with tf.Session() as session:
    init.run()
    for epoch in range(训练次数):
        for 批次 in range(mnist.train.num_examples // 批量):
            X_batch, y_batch = mnist.train.next_batch(批量)
            session.run(训练, feed_dict={X: X_batch, y: y_batch})
        训练集准确率 = 准确率.eval(feed_dict={X: X_batch, y: y_batch})
        验证集准确率 = 准确率.eval(feed_dict={X: mnist.validation.images, y: mnist.validation.labels})
        print(epoch, '训练集准确率:', 训练集准确率, '验证集准确率:', 验证集准确率)
    save_path = 存储器.save(session, '../data/II10/model/mnist_model.ckpt')

with tf.Session() as session:
    存储器.restore(session, '../data/II10/model/mnist_model.ckpt')
    X_new = mnist.test.images[:20]
    Z = 输出层.eval(feed_dict={X: X_new})
    y_pred = np.argmax(Z, axis=1)
    print('预测:', y_pred)
    print('实际:', mnist.test.labels[:20])

""" *** 用MNIST练习 *** """
reset_env()

输入数 = 28 * 28
输出数 = 10
学习率 = 0.01
X = tf.placeholder(tf.float32, shape=(None, 输入数), name='X')
y = tf.placeholder(tf.int64, shape=(None), name='y')
X_valid = mnist.validation.images
y_valid = mnist.validation.labels
with tf.name_scope('dnn'):
    隐藏层1 = tf.layers.dense(X, 300, name='hidden1', activation=tf.nn.relu)
    隐藏层2 = tf.layers.dense(隐藏层1, 100, name='hidden2', activation=tf.nn.relu)
    输出层 = tf.layers.dense(隐藏层2, 输出数, name='outputs')#未归一化概率
with tf.name_scope('loss'):
    交叉熵 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=输出层)
    损失 = tf.reduce_mean(交叉熵, name='loss')
    损失摘要格式 = tf.summary.scalar('loss', 损失)
with tf.name_scope('train'):
    优化器 = tf.train.GradientDescentOptimizer(学习率)
    训练 = 优化器.minimize(损失)
with tf.name_scope('eval'):
    正确否 = tf.nn.in_top_k(输出层, y, 1)
    准确率 = tf.reduce_mean(tf.cast(正确否, tf.float32))
    准确率摘要格式 = tf.summary.scalar('accuracy', 准确率)
init = tf.global_variables_initializer()

n_epochs = 1000
batch_size = 50
best_loss = np.infty
epochs_without_progress = 0
max_epochs_without_progress = 50

存储器 = tf.train.Saver()
日志记录器 = tf.summary.FileWriter('../data/II10/log/exerise/', tf.get_default_graph())
checkpoint_path = "../data/II10/model/exerise_model.ckpt"
checkpoint_epoch_path = checkpoint_path + ".epoch"
final_model_path = "../data/II10/model/final_model.ckpt"

with tf.Session() as session:
    if os.path.isfile(checkpoint_epoch_path):
        with open(checkpoint_epoch_path, "rb") as f:
            start_epoch = int(f.read())
        print("训练中断,从%d继续..."%start_epoch)
        存储器.restore(session, checkpoint_path)
    else:
        start_epoch = 0
        session.run(init)
    #可持久训练

    for epoch in range(start_epoch, n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            session.run(训练, feed_dict={X: X_batch, y: y_batch})
        验证集准确率, 验证集损失, 准确率摘要, 损失摘要 = session.run([准确率, 损失, 准确率摘要格式, 损失摘要格式], feed_dict={X: X_valid, y: y_valid})
        日志记录器.add_summary(准确率摘要, epoch)
        日志记录器.add_summary(损失摘要, epoch)
        if epoch % 5 == 0:
            print("Epoch:", epoch, "验证集准确率: {:.3f}%".format(验证集准确率 * 100), "验证集损失: {:.5f}".format(验证集损失))
            存储器.save(session, checkpoint_path)
            with open(checkpoint_epoch_path, "wb") as f:
                f.write(b"%d" % (epoch + 1))
            if 验证集损失 < best_loss:
                存储器.save(session, final_model_path)
                best_loss = 验证集损失
            else:
                epochs_without_progress += 5
                if epochs_without_progress > max_epochs_without_progress:
                    print("提前结束.")
                    break

os.remove(checkpoint_epoch_path)
X_test = mnist.test.images
y_test = mnist.test.labels
with tf.Session() as session:
    存储器.restore(session, final_model_path)
    练习准确率 = 准确率.eval(feed_dict={X: X_test, y: y_test})
print('练习准确率:', 练习准确率)
日志记录器.close()