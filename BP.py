'''
BP算法的执行流程：
手动设定神经网络的层数，每层的神经元的个数，学习率
随机初始化每条连接线权重和偏置
对于训练集中的每个输入x，输出y，先执行前向传输得到预测值
再根据真实值与预测值之间的误差执行逆向反馈
更新神经网络中每条连接线的权重和每层的偏置
重复以上过程
'''
'''
前向传输：
从输入层->隐藏层->输出层
'''
'''
逆向反馈：

'''
#导入必要的库文件
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

#激活函数及其导数的定义
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))


class Network(object):

    #对于神经网络，首先确定网络的尺寸，即几层，每层几个神经元
    def __init__(self, sizes):
        # 输入参数size为list类型，储存每层神经网络的神经元数目，长度表示层数
        self.num_layers = len(sizes)
        self.sizes = sizes
        # 除去输入层，随机产生每层中 y 个神经元的 biase 值（0 - 1）
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        # 随机产生每条连接线的 weight 值（0 - 1）
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
    #初始化之后进行的是前向传输
    def feedforward(self, a):
        #参数a：前一个神经元的输出或者是输入层的输入
        self.Z = []
        self.A = []
        self.A.append(a)
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, a) + b
        #缓存中间变量z
            self.Z.append(z)
            a = sigmoid(z)
            self.A.append(a)
        #对于a，只需要一个最后的输出即可
        return a
    
    #逆向反馈函数，输入为训练样本，输出为计算后的d_w、d_b
    def backprop(self, x, y):
        #初始化需要输出的d_w和d_b
        d_b = [np.zeros(b.shape) for b in self.biases]
        d_w = [np.zeros(w.shape) for w in self.weights]
        self.feedforward(x.T)
        delta = (self.A[-1]- y) * sigmoid_prime(self.Z[-1])
        d_b[-1] = delta
        d_w[-1] = np.dot(delta, self.A[-2].T)
        for j in range(2, self.num_layers):
            # 从倒数第 j 层开始更新
            z = self.Z[-j]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-j+1].T, delta) * sp
            d_b[-j] = delta
            d_w[-j] = np.dot(delta, self.A[-j-1].T)
        return (d_b, d_w)
    
    #根据计算出的d_w和d_b更新weights和biases，输入为学习率和训练样本
    def update_w_and_b(self, x, y, eta):
        #初始化需要输出的d_w和d_b
        B = [np.zeros(b.shape) for b in self.biases]
        W = [np.zeros(w.shape) for w in self.weights]
        d_b, d_w = self.backprop(x, y)
        B = [nb + d__b for nb, d__b in zip(B, d_b)]
        W = [nw + d__w for nw, d__w in zip(W, d_w)]
        self.weights = [w - eta * nw
                        for w, nw in zip(self.weights, W)]
        self.biases = [b - eta * nb
                       for b, nb in zip(self.biases, B)]

    #梯度下降
    def Gradient_Descent(self, x, y, eta, iteration):
        #eta为学习率
        for i in range(iteration):
            self.update_w_and_b(x, y, eta)
        
        
        
data = datasets.load_iris()
X = data['data']
y = (data.target != 0) * 1
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size = 0.5, random_state = 33) 
n = Network([4, 5, 3, 1])
n.Gradient_Descent(train_x, train_y, 0.01, 800)
y_prob = n.feedforward(test_x.T)