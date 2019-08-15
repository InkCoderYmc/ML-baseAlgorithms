#输入：样本集 D = {x1, x2, …… , xm};
#       低维空间维数d‘
#过程：
#   1：对所有样本进行中心化：xi <- xi - 1/m * sum(x1, x2, …… , xm)
#   2：计算样本的协方差矩阵XXT
#   3：对协方差矩阵做特征值分解
#   4：取最大的d’个特征值所对应的特征向量w1, w2, …… , wd'
#输出：投影矩阵W = {w1, w2, …… , wd'}

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets

def PCA(D, d_):
    #对样本进行中心化
    meanVals = np.mean(D, axis=0)  #求每一维度的均值，组成均值向量
    D_ = D - meanVals   #中心化
    #计算协方差，使用numpy中的cov函数
    XXT = np.cov(D_, rowvar=0) 
    #对协方差矩阵做特征值分解，通过eig函数得到特征值于特征向量
    eigVals, eigVects = np.linalg.eig(np.mat(XXT))
    #对特征值特征向量进行降序排序并取对应的前d’
    eigValInd = np.argsort(eigVals)   #argsort函数是默认的从小到大排序
    eigValInd = eigValInd[:-(d_+1):-1] #需要再加一步逆序，并顺带取前d’个特征值
    redEigVects = eigVects[:,eigValInd] #取对应的特征向量
    #计算特征矩阵
    D__ = D_ * redEigVects
    return D__

data=datasets.load_iris()
X=data['data']
y=data['target']
ax = Axes3D(plt.figure())
for c,i,target_name in zip('rgb',[0,1,2],data.target_names):
    ax.scatter(X[y==i ,0], X[y==i, 1], X[y==i,2], c=c, label=target_name)
ax.set_xlabel(data.feature_names[0])
ax.set_ylabel(data.feature_names[1])
ax.set_zlabel(data.feature_names[2])
ax.set_title("Iris")
plt.legend()
plt.show()

ax = plt.figure()
for c, i, target_name in zip("rgb", [0, 1, 2], data.target_names):
    plt.scatter(X[y == i, 0], X[y == i, 1], c=c, label=target_name)
plt.xlabel(data.feature_names[0])
plt.ylabel(data.feature_names[1])
plt.title("Iris")
plt.legend()
plt.show()


Z = PCA(X, 2)
X_reduce = Z.getA()

ax = plt.figure()
for c, i, target_name in zip("rgb", [0, 1, 2], data.target_names):
    plt.scatter(X_reduce[y == i, 0], X_reduce[y == i, 1], c=c, label=target_name)
plt.xlabel(data.feature_names[0])
plt.ylabel(data.feature_names[1])
plt.title("Iris")
plt.legend()
plt.show()

