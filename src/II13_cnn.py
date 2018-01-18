# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 14:06:58 2018
卷积神经网络
@author: Burning
"""

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def reset_env(seed = 42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)
#状态初始化函数,用来快速恢复ipython中的tensorflow到默认

def get_model_params():
    gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    return {gvar.op.name: value for gvar, value in zip(gvars, tf.get_default_session().run(gvars))}
#缓存训练过程中的参数,用于记录过程中的最佳参数

def restore_model_params(model_params):
    gvar_names = list(model_params.keys())
    assign_ops = {gvar_name: tf.get_default_graph().get_operation_by_name(gvar_name + "/Assign") for gvar_name in gvar_names}
    init_values = {gvar_name: assign_op.inputs[1] for gvar_name, assign_op in assign_ops.items()}
    feed_dict = {init_values[gvar_name]: model_params[gvar_name] for gvar_name in gvar_names}
    tf.get_default_session().run(assign_ops, feed_dict=feed_dict)
#从缓存字典中恢复参数,用于还原之前发现的最好参数

def cnn():
    mnist = input_data.read_data_sets("../data/mnist")
    height = 28
    width = 28
    channels = 1
    n_inputs = height * width
    conv1_fmaps = 12#卷积层卷积核种类(数目)
    conv1_ksize = 3#卷积核大小
    conv1_stride = 1#卷积核滑动步长
    conv1_pad = "SAME"#用0填补卷积核滑动中不能整除的部分
    conv2_fmaps = 16
    conv2_ksize = 3
    conv2_stride = 1
    conv2_pad = "SAME"
    conv2_dropout_rate = 0.25
    pool3_fmaps = conv2_fmaps
    n_fc4 = 32
    fc4_dropout_rate = 0.5
    n_outputs = 10
    
    with tf.name_scope('inputs'):
        X = tf.placeholder(tf.float32, shape=[None, n_inputs], name='X')
        X_re2d = tf.reshape(X, shape=[-1, height, width, channels], name='img')#将库中预处理的一维mnist还原成二维
        y = tf.placeholder(tf.int32, shape=[None], name='y')
        training = tf.placeholder_with_default(False, shape=[], name='training')#指示训练还是测试状态
    
    with tf.name_scope('cnn'):
        conv1 = tf.layers.conv2d(X_re2d, filters=conv1_fmaps, kernel_size=conv1_ksize, strides=conv1_stride, padding=conv1_pad, activation=tf.nn.relu, name='conv1')
        conv2 = tf.layers.conv2d(conv1, filters=conv2_fmaps, kernel_size=conv2_ksize, strides=conv2_stride, padding=conv2_pad, activation=tf.nn.relu, name='conv2')
        pool3 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID', name='pool3')
        pool3_flat = tf.reshape(pool3, shape=[-1, pool3_fmaps * 14 * 14], name='pool3_flat')
        pool3_drop = tf.layers.dropout(pool3_flat, conv2_dropout_rate, training=training)
        fc4 = tf.layers.dense(pool3_drop, units=n_fc4, activation=tf.nn.relu, name='fc4')
        fc4_drop = tf.layers.dropout(fc4, rate=fc4_dropout_rate, training=training)
    
    with tf.name_scope('outputs'):
        logits = tf.layers.dense(fc4, units=n_outputs, name='output')
        Y_proba = tf.nn.softmax(logits, name='Y_proba')
    
    with tf.name_scope('train'):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
        loss = tf.reduce_mean(xentropy)
        optimizer = tf.train.AdamOptimizer()
        training_op = optimizer.minimize(loss)
    
    with tf.name_scope('eval'):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    
    init = tf.global_variables_initializer()
    n_epochs = 1000
    batch_size = 20
    best_loss_val = np.infty
    check_interval = 500#验证效果间隔
    checks_since_last_progress = 0
    max_checks_without_progress = 20
    best_model_params = None#最佳参数字典
    
    with tf.Session() as session:
        init.run()
        for epoch in range(n_epochs):
            for iteration in range(mnist.train.num_examples // batch_size):
                X_batch, y_batch = mnist.train.next_batch(batch_size)
                session.run(training_op, feed_dict={X: X_batch, y: y_batch, training: True})
                if iteration % check_interval == 0:
                    loss_val = loss.eval(feed_dict={X: mnist.validation.images, y: mnist.validation.labels})
                    if loss_val < best_loss_val:
                        best_loss_val = loss_val
                        checks_since_last_progress = 0
                        best_model_params = get_model_params()
                    else:
                        checks_since_last_progress += 1
            acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
            acc_valid = accuracy.eval(feed_dict={X: mnist.validation.images, y: mnist.validation.labels})
            print('Epoch %d, train acc: %.4f%%, vaild acc: %.4f%%, best_loss: %.6f' % (epoch, acc_train*100, acc_valid*100, best_loss_val))
            if checks_since_last_progress > max_checks_without_progress:
                print('提前结束!')
                break
        if best_model_params:
            restore_model_params(best_model_params)
        acc_test = accuracy.eval(feed_dict={X: mnist.test.images, y: mnist.test.labels})
        print ('最终测试正确率: %.4f%%' %(acc_test * 100))
#CNN在MNIST的样例,笔记本CPU+4G运行困难