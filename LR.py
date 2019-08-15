#将线性回归扩展到分类情况中
#对于两类，类0和类1，只需要指定这两类之一的概率，因为两者概率和为1
#使用logistic sigmoid函数将线性函数的输出压缩进区间（0， 1）
#该值可以解释为概率 p(y=1|x;θ)=σ（θT*x）
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn import datasets



#定义sigmoid函数g(z)=1/(1+e^(-z))
def sigmoid_function(z):
    return 1 / (1 + np.exp(-z))

#定义代价函数cost(h(x), y)=-y*log(h(x))-(1-y)log(1-h(x))
def cost_function(h, y):
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean() #这里的mean()函数等价于1/m

#
def logistic_regression(alpha, X, Y):
    theta = np.zeros(X.shape[1]) #初始化theta
    m = y.size
    z = np.dot(X, theta) #得到初始化的z
    h = sigmoid_function(z) #得到初始化的h函数
    J = cost_function(h, y) #初始化J
    #循环直到收敛
    while True:
        gradient = 1 / m * np.dot(X.T, h - y) #计算偏导数
        theta = theta - alpha * gradient #根据学习率更新theta
        z = np.dot(X, theta) #更新z
        h = sigmoid_function(z) #更新h
        deltaJ = J
        J = cost_function(h, y) #更新J
        deltaJ = math.fabs(J - deltaJ)
        if deltaJ < 1e-8: #这里用1e-8为标准来衡量是否收敛
            break
    return theta

def predict_prob(X): #预测函数
    return sigmoid_function(np.dot(X, theta))



if __name__ == '__main__':
    iris = datasets.load_iris()
    X = iris.data[:, :2]
    y = (iris.target != 0) * 1

    alpha = 0.1
    theta = logistic_regression(alpha, X, y)
    print (theta)

    plt.figure(figsize=(10, 6))
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='b', label='0')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='r', label='1')
    (x1_min, x1_max) = (X[:, 0].min(), X[:, 0].max())
    (x2_min, x2_max) = (X[:, 1].min(), X[:, 1].max())
    (xx1, xx2) = np.meshgrid(np.linspace(x1_min, x1_max),
                             np.linspace(x2_min, x2_max))
    grid = np.c_[xx1.ravel(), xx2.ravel()]
    probs = predict_prob(grid).reshape(xx1.shape)
    plt.contour(
        xx1,
        xx2,
        probs,
        [0.5],
        linewidths=1,
        colors='black',
        )

    plt.legend()
