# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 12:53:25 2018
tensorflow的Helloworld,附带线性规划练习
@author: Burning
"""

""" *** 初始化 *** """
import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from datetime import datetime

def reset_env(seed = 42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)
#用来让某部分的随机化效果和参考书一致,每个部分都应当用这个函数先初始化

""" *** Helloworld,构建第一个计算图 *** """
x = tf.Variable(3, name='x')#tensorflow暂时还不支持中文name
y = tf.Variable(4, name='y')
f = x * x * y + y + 2

session = tf.Session()
session.run(x.initializer)#变量定义完了还要初始化
session.run(y.initializer)
result = session.run(f)
print('Hi', result)
session.close()

with tf.Session() as session:
    x.initializer.run()#=>tf.get_default_session().run(x.initializer)
    y.initializer.run()
    result = f.eval()#=>tf.get_default_session().run(f)
    print('Hi', result)
#这种写法相当于用with指定默认tensorflow会话,其中每行指令在默认会话基础上进行,with结束后会自动close会话

初始化节点=tf.global_variables_initializer()
with tf.Session() as session:
    初始化节点.run()
    result = f.eval()
    print('Hi', result)
#可以构建一个初始化节点来完成所有变量初始化,注意初始化节点定义后必须要运行一次才能完成初始化。

#session = tf.Session()
#初始化节点.run()
#result = f.eval()
#print(result)
#session.close()
#这种方法只能在交互终端使用

""" *** 管理计算图 *** """
x1 = tf.Variable(1)
print(x1.graph is tf.get_default_graph())
graph = tf.Graph()
with graph.as_default():
    x2 = tf.Variable(2)
print(x2.graph is graph)
print(x2.graph is tf.get_default_graph)
#is判断左右两个变量是否引用为同一份数据,节点会默认添加到当前默认计算图

""" *** 节点生命周期 *** """
w = tf.constant(3)
x = w + 2
y = x + 5
z = x * 3
#常数和初始变量会存在于整个会话,其他节点的说一句会在每次run或eval后被释放

with tf.Session() as session:
    print('第一遍计算依赖', y.eval())
    print('第二遍计算依赖', z.eval())
with tf.Session() as session:
    y_val, z_val = session.run([y, z])
    print('只计算一次依赖', y_val, z_val)
#为了更高效的计算,不应该在一张图分次计算不同的值,即选择第二种写法

""" *** 线性回归样例(加利福尼亚住房数据集) *** """
数据集 = fetch_california_housing(data_home='../data/II09_tensorflow')
样本数, 特征数 = 数据集.data.shape
加工数据集 = np.c_[np.ones((样本数, 1)), 数据集.data]#加个1便于统一计算b

#使用正规方程,其实就是求导令导数为0得出回归系数的解
X = tf.constant(加工数据集, dtype=tf.float32, name='X')
y = tf.constant(数据集.target.reshape(-1,1), dtype=tf.float32, name='y')
XT = tf.transpose(X)
回归系数 = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)#损失函数导数为0时W=(X.T*X)^-1*X.T*Y
with tf.Session():
    回归系数tf估计 = 回归系数.eval()
print('tensorflow 计算的回归系数:', 回归系数tf估计)
X = np.mat(加工数据集)
y = np.mat(数据集.target.reshape(-1,1))
回归系数np估计 = (X.T * X).I * X.T * y
print('numpy 计算的回归系数:', 回归系数np估计)
#tensorflow和numpy算出来的结果由约1%的不同

""" *** 使用批量(离线)梯度下降法 *** """
训练次数 = 1000
学习率 = 0.01
scaler = StandardScaler()#标准归一化器,将数据减去均值除以方差
归一化数据集 = scaler.fit_transform(数据集.data)#梯度下降需要归一化数据
加工归一化数据集 = np.c_[np.ones((样本数, 1)), 归一化数据集]
X = tf.constant(加工归一化数据集, dtype=tf.float32, name='X')
y = tf.constant(数据集.target.reshape(-1,1), dtype=tf.float32, name='y')
回归系数 = tf.Variable(tf.random_uniform([特征数+1, 1], -1.0, 1.0, seed=42), name='theta')#不锁定seed,结果会不一样
y_pred = tf.matmul(X, 回归系数, name='predictions')
误差 = y_pred - y
均方误差 = tf.reduce_mean(tf.square(误差), name='mse')

#手动计算梯度
梯度 = 2 / 样本数 * tf.matmul(tf.transpose(X), 误差)
训练结果 = tf.assign(回归系数, 回归系数 - 学习率 * 梯度)
初始化节点 = tf.global_variables_initializer()
with tf.Session() as session:
    session.run(初始化节点)
    for epoch in range(训练次数):
        if epoch % 100 == 0:
            print('Epoch', epoch, '均方误差 =', 均方误差.eval())
        session.run(训练结果)
    print('手动计算梯度得到回归系数:', 回归系数.eval())

#使用自动微分
梯度 = tf.gradients(均方误差, [回归系数])[0]
训练结果 = tf.assign(回归系数, 回归系数 - 学习率 * 梯度)
初始化节点 = tf.global_variables_initializer()
with tf.Session() as session:
    session.run(初始化节点)
    for epoch in range(训练次数):
        if epoch % 100 == 0:
            print('Epoch', epoch, '均方误差 =', 均方误差.eval())
        session.run(训练结果)
    print('使用自动微分得到归回系数:', 回归系数.eval())

#使用梯度下降优化方法
优化器 = tf.train.GradientDescentOptimizer(learning_rate=学习率)
训练结果 = 优化器.minimize(均方误差)
初始化节点 = tf.global_variables_initializer()
with tf.Session() as session:
    session.run(初始化节点)
    for epoch in range(训练次数):
        if epoch % 100 == 0:
            print('Epoch', epoch, '均方误差 =', 均方误差.eval())
        session.run(训练结果)
    print('使用梯度下降优化器得到回归系数:', 回归系数.eval())

##使用动量优化方法
优化器 = tf.train.MomentumOptimizer(learning_rate = 学习率, momentum=0.9)
训练结果 = 优化器.minimize(均方误差)
初始化节点 = tf.global_variables_initializer()
with tf.Session() as session:
    session.run(初始化节点)
    for epoch in range(训练次数):
        if epoch % 100 == 0:
            print('Epoch', epoch, '均方误差 =', 均方误差.eval())
        session.run(训练结果)
    print('使用动量优化器得到回归系数:', 回归系数.eval())

""" *** 使用数据流训练 *** """
A = tf.placeholder(tf.float32, shape=(None, 3))
B = A + 5
with tf.Session() as session:
    B_v1 = B.eval(feed_dict={A: [[1,2,3]]})#每次输入的feed_dict必须是二维，且列数同设置
    B_v2 = B.eval(feed_dict={A: [[4,5,6],[7,8,9]]})
print(B_v1, B_v2)

#用小批量数据训练之前的梯度下降优化器
学习率 = 0.01
训练次数 = 10
批量 = 100
批次 = int(np.ceil(样本数 / 批量))#对于随机取小批数据，课随意取值
X = tf.placeholder(tf.float32, shape=(None, 特征数+1), name='X')
y = tf.placeholder(tf.float32, shape=(None, 1), name='y')
回归系数 = tf.Variable(tf.random_uniform([特征数+1, 1], -1.0, 1.0, seed=42), name='theta')#不锁定seed,结果会不一样
y_pred = tf.matmul(X, 回归系数, name='predictions')
误差 = y_pred - y
均方误差 = tf.reduce_mean(tf.square(误差), name='mse')
优化器 = tf.train.GradientDescentOptimizer(learning_rate=学习率)
训练结果 = 优化器.minimize(均方误差)
初始化节点 = tf.global_variables_initializer()

def 取一批数据(批量):
    切片索引 = np.random.randint(样本数, size=批量)
    X_batch = 加工归一化数据集[切片索引]
    y_batch = 数据集.target.reshape(-1,1)[切片索引]
    return X_batch, y_batch

with tf.Session() as session:
    session.run(初始化节点)
    for epoch in range(训练次数):
        for batch_index in range(批次):
            X_batch, y_batch = 取一批数据(批量)
            session.run(训练结果, feed_dict={X: X_batch, y: y_batch})
    print('流式训练出的回归系数', 回归系数.eval())

""" *** 存取模型 *** """
训练次数 = 1000
学习率 = 0.01
scaler = StandardScaler()#标准归一化器,将数据减去均值除以方差
归一化数据集 = scaler.fit_transform(数据集.data)#梯度下降需要归一化数据
加工归一化数据集 = np.c_[np.ones((样本数, 1)), 归一化数据集]
X = tf.constant(加工归一化数据集, dtype=tf.float32, name='X')
y = tf.constant(数据集.target.reshape(-1,1), dtype=tf.float32, name='y')
回归系数 = tf.Variable(tf.random_uniform([特征数+1, 1], -1.0, 1.0, seed=42), name='theta')#不锁定seed,结果会不一样
y_pred = tf.matmul(X, 回归系数, name='predictions')
误差 = y_pred - y
均方误差 = tf.reduce_mean(tf.square(误差), name='mse')
优化器 = tf.train.GradientDescentOptimizer(learning_rate=学习率)
训练结果 = 优化器.minimize(均方误差)
初始化节点 = tf.global_variables_initializer()
#以上代码和使用梯度下降优化器的离线算法构造一致

存储器 = tf.train.Saver()#应该在所有节点之后定义存储器
with tf.Session() as session:
    session.run(初始化节点)
    for epoch in range(训练次数):
        if epoch % 100 == 0:
            print('Epoch', epoch, '均方误差 =', 均方误差.eval())
        session.run(训练结果)
    print('得到回归系数并存储会话:', 回归系数.eval())
    save_path = 存储器.save(session, '../data/II09_tensorflow/tmp_model.ckpt')
with tf.Session() as session:
    存储器.restore(session, save_path)
    print('从存储器恢复会话重估回归系数', 回归系数.eval())


存储器 = tf.train.Saver({"weights": 回归系数})#可以这样指定只以某个名字保存某个节点状态
存储器 = tf.train.import_meta_graph("../data/II09_tensorflow/tmp_model.ckpt.meta")#存储器默认会单独存储一个计算图文件，读取计算图文件可以将缓存的计算图恢复并得到一个存储器

""" *** 可视化计算图(TensorBoard) *** """
时刻 = datetime.utcnow().strftime('%Y%m%d%H%M%S')
日志根路径 = '../data/II09_tensorflow/tf_logs'
日志子路径 = '{}/run-{}'.format(日志根路径, 时刻)
#设置用于展示的日志文件位置

学习率 = 0.01
训练次数 = 10
批量 = 100
批次 = int(np.ceil(样本数 / 批量))#对于随机取小批数据，课随意取值
X = tf.placeholder(tf.float32, shape=(None, 特征数+1), name='X')
y = tf.placeholder(tf.float32, shape=(None, 1), name='y')
回归系数 = tf.Variable(tf.random_uniform([特征数+1, 1], -1.0, 1.0, seed=42), name='theta')#不锁定seed,结果会不一样
y_pred = tf.matmul(X, 回归系数, name='predictions')
误差 = y_pred - y
均方误差 = tf.reduce_mean(tf.square(误差), name='mse')
优化器 = tf.train.GradientDescentOptimizer(learning_rate=学习率)
训练结果 = 优化器.minimize(均方误差)
初始化节点 = tf.global_variables_initializer()
#和流式训练部分一样

均方误差摘要 = tf.summary.scalar('MSE', 均方误差)#tensorflow内部名称暂不能写中文
日志记录器 = tf.summary.FileWriter(日志子路径, tf.get_default_graph())
with tf.Session() as session:
    session.run(初始化节点)
    for epoch in range(训练次数):
        for batch_index in range(批次):
            X_batch, y_batch = 取一批数据(批量)
            if batch_index % 10 == 0:
                摘要数据 = 均方误差摘要.eval(feed_dict={X: X_batch, y: y_batch})
                step = epoch * 批量 + batch_index
                日志记录器.add_summary(摘要数据, step)
            session.run(训练结果, feed_dict={X: X_batch, y: y_batch})
    print('流式训练并记录回归系数变化', 回归系数.eval())
日志记录器.close()
#使用命令  tensorboard --logdir 日志根路径
#在按照提示打开网页即可查看tensorboard

""" *** 命名空间 *** """
a1 = tf.Variable(0, name="a")      # name == "a"
a2 = tf.Variable(0, name="a")      # name == "a_1"
with tf.name_scope("param"):       # name == "param"
    a3 = tf.Variable(0, name="a")  # name == "param/a"
with tf.name_scope("param"):       # name == "param_1"
    a4 = tf.Variable(0, name="a")  # name == "param_1/a"
for node in (a1, a2, a3, a4):
    print(node.op.name)
#tensorflow中会对重复使用的命名空间加编号

""" *** 模块化 *** """
def relu(x):
    with tf.name_scope('relu'):
        w = tf.Variable(tf.random_normal((int(x.get_shape()[1]), 1)), name='weights')
        b = tf.Variable(0.0, name='bias')
        z = tf.add(tf.matmul(x, w), b, name='z')
        return tf.maximum(z, 0, name='max')
特征数 = 3
X = tf.placeholder(tf.float32, shape=(None, 特征数), name='X')
relus = [relu(X) for i in range(5)]#调用构造函数快速构造相同结构模块
output = tf.add_n(relus, name='output')
日志记录器 = tf.summary.FileWriter("../data/II09_tensorflow/tf_logs/relu2", tf.get_default_graph())
日志记录器.close()
#使用命令  tensorboard --logdir 日志根路径
#在按照提示打开网页即可通过tensorboard查看网络结构

""" *** 共享变量 *** """
#多种写法,具体看书,总之都要先建立后复用,没dict.get方便