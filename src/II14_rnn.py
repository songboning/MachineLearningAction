# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 17:31:32 2018
循环神经网络
@author: Burning
"""

""" *** 初始化 *** """
import tensorflow as tf
import numpy as np

def reset_env(seed = 42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)
#用来让某部分的随机化效果和参考书一致,每个部分都应当用这个函数先初始化

""" *** 朴素循环神经网络 *** """
n_inputs = 3
n_neurons = 5
n_steps = 3#神经元循环次数
X0_batch = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 0, 1]])
X1_batch = np.array([[9, 8, 7], [0, 0, 0], [6, 5, 4], [3, 2, 1]])
X2_batch = np.array([[0, 1, 2], [3, 4, 5], [0, 0, 0], [9, 8, 7]])
X_batch = np.array([X0_batch, X1_batch, X2_batch])

#长为3的序列输入到5个神经元组成的一个RNN单元,只执行三轮,手动实现计算图,无训练只是演示RNN运行
reset_env()
X0 = tf.placeholder(tf.float32, [None, n_inputs])#第一批输入
X1 = tf.placeholder(tf.float32, [None, n_inputs])#第二批输入
X2 = tf.placeholder(tf.float32, [None, n_inputs])#第三批输入
#每个时刻可以输入一批,每批shape要一直
Wx = tf.Variable(tf.random_normal(shape=[n_inputs, n_neurons],dtype=tf.float32))
Wy = tf.Variable(tf.random_normal(shape=[n_neurons,n_neurons],dtype=tf.float32))
b = tf.Variable(tf.zeros([1, n_neurons], dtype=tf.float32))
#神经元内的运算,将新输入和上次输出都处理成4*5的格式
Y0 = tf.tanh(tf.matmul(X0, Wx) + b)#第一次输出
Y1 = tf.tanh(tf.matmul(Y0, Wy) + tf.matmul(X1, Wx) + b)#第二次输出
Y2 = tf.tanh(tf.matmul(Y1, Wy) + tf.matmul(X2, Wx) + b)#第三次输出
init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    Y0_val, Y1_val, Y2_val = sess.run([Y0, Y1, Y2], feed_dict={X0: X0_batch, X1: X1_batch, X2: X2_batch})
print('Y0:', Y0_val, '\nY1:', Y1_val, '\nY2:', Y2_val)

#使用静态展开,将之前的计算图定义自动展开,这样节省python代码,但计算图还是一样很大
reset_env()
X = tf.placeholder(tf.float32, [n_steps, None, n_inputs])
输入序列 = tf.unstack(X)
基本神经元 = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
输出序列, 最终状态 = tf.contrib.rnn.static_rnn(cell=基本神经元, inputs=输入序列, dtype=tf.float32)#朴素RNN的最终状态就是最后一个输出
Y = tf.stack(输出序列)
init = tf.global_variables_initializer()
with tf.Session() as session:
    init.run()
    Y_val = Y.eval(feed_dict={X: X_batch})
print('静态展开:', Y_val)
#原书中改变了维度顺序,可能是为了将batch统一放在第一个

#使用动态RNN,节省开销,真正循环执行
reset_env()
X = tf.placeholder(tf.float32, [n_steps, None, n_inputs])#如果不将batch_size的None放在第一维度,就要将动态展开的time_major设为True表示第一位是循环次数
基本神经元 = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
输出序列, 最终状态 = tf.nn.dynamic_rnn(cell=基本神经元, inputs=X, dtype=tf.float32, time_major=True)#time_major默认False表示输入的第一维度是batch_size
init = tf.global_variables_initializer()
with tf.Session() as session:
    init.run()
    Y_val = 输出序列.eval(feed_dict={X: X_batch})
print('动态展开:', Y_val)

##设置序列长度
L_batch = np.array([2, 1, 3, 2])#batch中每个的序列长度,也就是循环次数
reset_env()
X = tf.placeholder(tf.float32, [n_steps, None, n_inputs])
序列长度 = tf.placeholder(tf.int32, [None])
基本神经元 = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
输出序列, 最终状态 = tf.nn.dynamic_rnn(cell=基本神经元, inputs=X, dtype=tf.float32, sequence_length=序列长度, time_major=True)
init = tf.global_variables_initializer()
with tf.Session() as session:
    init.run()
    outputs_val, states_val = session.run(
        [输出序列, 最终状态], feed_dict={X: X_batch, 序列长度: L_batch})
print('限制长度', outputs_val)
#可见每批数据是分开运行的,是多个实例

""" *** 训练浅层循环神经网络 *** """
reset_env()
n_steps = 28
n_inputs = 28
n_neurons = 150#每层神经网络150个神经元
n_layers = 3#每个循环单元有三层神经网络构成
n_outputs = 10
学习率 = 0.001

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../data/mnist")
X_test = mnist.test.images.reshape((-1, n_steps, n_inputs))
y_test = mnist.test.labels
#按照tensorflow提供的定制方式使用mnist数据集

X = tf.placeholder(tf.float32, shape=[None, n_steps, n_inputs], name='X')#使用默认的batch_size为第一维度
y = tf.placeholder(tf.int32, shape=[None], name='y')
基本神经元们 = [tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu) for layer in range(n_layers)]
多层循环单元 = tf.contrib.rnn.MultiRNNCell(基本神经元们)#使用多层神经网络作为循环单元
多层输出序列, 多层最终状态 = tf.nn.dynamic_rnn(cell=多层循环单元, inputs=X, dtype=tf.float32, time_major=False)
最终状态 = tf.concat(axis=1, values=多层最终状态)#对多层循环单元的多层状态合并成一个张量
logits = tf.layers.dense(最终状态, n_outputs, name='logits')#只取最后一次输出,,就在这一步将outputs的复杂输出处理成需要的格式,比如这里只取最后一次输出聚成10个点,或者序列回归时将outputs分别整理成单个输出
xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits, name='xentropy')#交叉熵
loss = tf.reduce_mean(xentropy, name='loss')#以最后一个单元输出结果和标签的交叉熵为损失
optimizer = tf.train.AdamOptimizer(learning_rate=学习率)
training_op = optimizer.minimize(loss, name='train')
correct = tf.nn.in_top_k(predictions=logits, targets=y, k=1, name='judge')
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='acc')
init = tf.global_variables_initializer()

n_epochs = 10
batch_size = 150
with tf.Session() as session:
    init.run()
    for epoch in range(n_epochs):
        for i in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            X_batch = X_batch.reshape((-1, n_steps, n_inputs))
            session.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
        print(epoch, "训练准确率:", acc_train, "测试准确率:", acc_test)