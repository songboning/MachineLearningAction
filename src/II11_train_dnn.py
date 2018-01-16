# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 17:09:56 2018
训练深度神经网络
@author: Burning
"""

""" *** 初始化 *** """
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.exceptions import NotFittedError

def reset_env(seed = 42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)
#用来让某部分的随机化效果和参考书一致,每个部分都应当用这个函数先初始化

#处理所需数据
mnist = input_data.read_data_sets("../data/mnist")
X_train1 = mnist.train.images[mnist.train.labels < 5]
y_train1 = mnist.train.labels[mnist.train.labels < 5]
X_valid1 = mnist.validation.images[mnist.validation.labels < 5]
y_valid1 = mnist.validation.labels[mnist.validation.labels < 5]
X_test1 = mnist.test.images[mnist.test.labels < 5]
y_test1 = mnist.test.labels[mnist.test.labels < 5]

#定义所需要的额外的激活函数和参数初始化方法
def leaky_relu(alpha=0.01):
    def parametrized_leaky_relu(z, name=None):
        return tf.maximum(alpha * z, z, name=name)
    return parametrized_leaky_relu
#leaky_relu激活函数的自定义(目前tensorflow中没有)

def selu(scale=1.0507009873554804934193349852946, alpha=1.6732632423543772848170429916717):
    def parametrized_selu(z, name=None):
        return scale * tf.where(z >= 0.0, z, alpha * tf.nn.elu(z))
    return parametrized_selu
#2017新提出的激活函数

he_init = tf.contrib.layers.variance_scaling_initializer()
#一种适合RELU激活的权值初始化方法,主要提供值域和方差参数

""" *** 深度学习 *** """
class DNNClassifier:
    def __init__(self, 隐藏层数=5, 每层神经元数=160, 优化方法=tf.train.AdamOptimizer, 学习率=0.01, 批量=100, 激活函数=tf.nn.relu, 参数初始化方法=he_init, 活性=0.5, 归一化趋势=0.98, 随机种子=None):
        self.n_hidden_layers = 隐藏层数
        self.n_neurons = 每层神经元数
        self.optimizer_class = 优化方法
        self.learning_rate = 学习率
        self.batch_size = 批量
        self.activation = 激活函数
        self.initializer = 参数初始化方法
        self.dropout_rate = 活性
        self.batch_norm_momentum = 归一化趋势
        self.random_state = 随机种子
        self._session = None#前导一个下划线表示不应从外部调用
        self._training = None#用来为dropout和batchnorm表示是否再训练状态
        self._graph = None
        self.classname = None
        self.classindex = None
    #初始化配置超参数

    def _dnn(self, inputs):
        for layer_no in range(self.n_hidden_layers):
            if self.dropout_rate:
                inputs = tf.layers.dropout(inputs, rate=self.dropout_rate, training=self._training, name='dropout')
            inputs = tf.layers.dense(inputs, self.n_neurons, kernel_initializer=self.initializer, name='hidden%d'%layer_no)
            if self.batch_norm_momentum:
                inputs = tf.layers.batch_normalization(inputs, momentum=self.batch_norm_momentum, training=self._training)
            inputs = self.activation(inputs, name='hidden%d_out'%layer_no)
        return inputs
    #批量构建隐含层,并支持dropout和batchnorm正则化

    def _build_graph(self, n_inputs, n_outputs):
        if self.random_state is not None:
            tf.set_random_seed(self.random_state)
            np.random.seed(self.random_state)
        #初始化随机种子
        X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
        y = tf.placeholder(tf.int32, shape=(None), name='y')
        if self.dropout_rate or self.batch_norm_momentum:
            self._training = tf.placeholder_with_default(False, shape=(), name='training')
        else:
            self._training = None
        #初始化训练状态占位
        dnn_outputs = self._dnn(X)
        logits = tf.layers.dense(dnn_outputs, n_outputs, kernel_initializer=self.initializer, name='logits')
        Y_proba = tf.nn.softmax(logits, name='Y_probability')#各个结果可能性
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits, name='xentropy')#交叉熵
        loss = tf.reduce_mean(xentropy, name='loss')#损失
        optimizer = self.optimizer_class(learning_rate=self.learning_rate)#优化器,有默认name
        training_op = optimizer.minimize(loss)#训练操作
        correct = tf.nn.in_top_k(predictions=logits, targets=y, k=1, name='judge')#给出预测是否正确
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='accuracy')
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        self._node_X = X
        self._node_y = y
        self._node_Y_proba = Y_proba
        self._node_loss = loss
        self._node_training_op = training_op
        self._node_accuracy = accuracy
        self._node_init = init
        self._saver = saver
    #构建深度神经网络,并配置内部变量

    def close_session(self):
        if self._session:
            self._session.close()

    def _get_model_params(self):
        with self._graph.as_default():
            gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        return {gvar.op.name: value for gvar, value in zip(gvars, self._session.run(gvars))}
    #得到所有tf变量的值(用来提前结束避免过拟合,比存盘快)

    def _restore_model_params(self, model_params):
        gvar_names = list(model_params.keys())
        assign_ops = {gvar_name: self._graph.get_operation_by_name(gvar_name + '/Assign') for gvar_name in gvar_names}
        init_values = {gvar_name: assign_op.inputs[1] for gvar_name, assign_op in assign_ops.items()}
        feed_dict = {init_values[gvar_name]: model_params[gvar_name] for gvar_name in gvar_names}
        self._session.run(assign_ops, feed_dict=feed_dict)
    #设置所有变量(用于提前结束避免过拟合, 比度盘快)

    def fit(self, X, y, n_epochs=100, X_valid=None, y_valid=None):
        self.close_session()
        n_inputs = X.shape[1]
        self.classname = np.unique(y)#记录类别名称
        n_outputs = len(self.classname)
        self.classindex = {label: index for index, label in enumerate(self.classname)}#类别编号
        y = np.array([self.classindex[label] for label in y], dtype=np.int32)
        self._graph = tf.Graph()
        with self._graph.as_default():
            self._build_graph(n_inputs, n_outputs)
            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        max_checks_without_progress = 20
        checks_without_progress = 0
        best_loss = np.infty
        best_params = None
        self._session = tf.Session(graph=self._graph)
        with self._session.as_default() as session:
            self._node_init.run()
            for epoch in range(n_epochs):
                rand_index = np.random.permutation(len(X))
                for rand_indices in np.array_split(rand_index, len(X) // self.batch_size):
                    X_batch, y_batch = X[rand_indices], y[rand_indices]
                    feed_dict = {self._node_X: X_batch, self._node_y: y_batch}
                    if self._training is not None:
                        feed_dict[self._training] = True
                    session.run(self._node_training_op, feed_dict=feed_dict)
                    if extra_update_ops:
                        session.run(extra_update_ops, feed_dict=feed_dict)
                if X_valid is not None and y_valid is not None:
                    loss_val, acc_val = session.run([self._node_loss, self._node_accuracy], feed_dict={self._node_X: X_valid, self._node_y: y_valid})
                    if loss_val < best_loss:
                        best_params = self._get_model_params()
                        best_loss = loss_val
                        checks_without_progress = 0
                    else:
                        checks_without_progress += 5
                    print('%d\tBest Loss: %.6f\tValid Loss: %.6f\tAccuracy: %.2f%%' %(epoch, best_loss, loss_val, acc_val*100))
                    if checks_without_progress > max_checks_without_progress:
                        print('提前结束!')
                        break
                else:
                    loss_train, acc_train = session.run([self._node_loss, self._node_accuracy], feed_dict={self._node_X: X_batch, self.node_y: y_batch})
                    print('%d\tBest Loss: %.6f\tTrain Batch Loss: %.6f\tAccuracy: %.2f%%' %(epoch, best_loss, loss_train, acc_train*100))
            if best_params:
                self._restore_model_params(best_params)#还原最佳状态
        return self
    #用训练数据拟合深度神经网络

    def predict_probability(self, X):
        if not self._session:
            raise NotFittedError('The instance %s is not fitted yet' %self.__class__.__name__)
        with self._session.as_default():
            return self._node_Y_proba.eval(feed_dict={self._node_X: X})
    #预测标签可能性

    def predict(self, X):
        class_indices = np.argmax(self.predict_probability(X), axis=1)
        return np.array([self.classname[class_index] for class_index in class_indices], dtype=np.int32)
    #得到最可能单一预测值

    def save(self, path):
        self._saver.save(self._session, save_path=path)
#MLP深度神经网络分类器,默认参数是用下面候选参数调参后选出的最优参数

dnn_clf = DNNClassifier(随机种子=42)
dnn_clf.fit(X_train1, y_train1, n_epochs=1000, X_valid=X_valid1, y_valid=y_valid1)
y_pred = dnn_clf.predict(X_test1)
print('测试结果:', sum(y_pred == y_test1) / len(y_test1))

候选超参数 = {
        '隐藏层数': [5],
        '每层神经元数': [10, 30, 50, 70, 90, 100, 120, 140, 160],
        '优化方法': [tf.train.AdamOptimizer],
        '学习率': [0.01, 0.02, 0.05, 0.1],
        '批量': [10, 50, 100, 500],
        '激活函数': [tf.nn.relu, tf.nn.elu, leaky_relu(alpha=0.01), selu()],
        '参数初始化方法': [he_init],
        '活性': [None, 0.2, 0.3, 0.4, 0.5, 0.6],
        '归一化趋势': [None, 0.9, 0.95, 0.98, 0.99, 0.999]
    }
#这里列出了可调超参数的一些候选,重组它们也许能得到更好的效果,当然也可以用sklearn中的RandomizedSearchCV来较快速的自动调参
