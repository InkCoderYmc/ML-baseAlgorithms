
#导入必要库
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.datasets import load_iris

#计算距离
def distance_computer(x, y):
    return np.sqrt(np.sum((x-y)**2))

#k-means函数
def KMEANS(datat_input, k):

    #初始化均值向量，随机选择k个样本
    m,n = datat_input.shape
    mu = np.zeros((k,n)) #均值向量μ
    for i in range(k):
        index = int(np.random.uniform(0,m))
        mu[i,:] = datat_input[index,:]

    #定义一个二维矩阵用于存储对应样本的所属簇
    C = np.mat(np.zeros((m,2)))
    flag = True
    while flag:
        flag = False
        for j in range(m):
            #初始化最短距离和簇号
            minDistance = 10000.00
            minIndex = -1

            for i in range(k):
                distance = distance_computer(mu[i,:], datat_input[j,:])
                if distance < minDistance:
                    minDistance = distance
                    minIndex = i

            #更新每个样本对应的簇
            if C[j, 0] != minIndex:
                flag = True
                C[j,:] =  minIndex, minDistance

        #更新均值向量
        for i in range(k):
            points = datat_input[np.nonzero(C[:,0].A == i)[0]]  # 获取簇类所有的点
            mu[i,:] = np.mean(points ,axis=0)   # 对矩阵的行求均值

    return mu, C

iris = load_iris()
X = iris.data[:,2:] 
#绘制数据分布图
plt.scatter(X[:, 0], X[:, 1], c = "red", marker='o', label='see')
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend(loc=2)
plt.show()

k = 3
mu, C = KMEANS(X, k)
m,n = X.shape
C = C.astype(np.int32)
D = [0] * m
for i in range(m):
    D[i] = C[i, 0]
x0=[]
x1=[]
x2=[]
for i in range(m):
    if(D[i]==0):
        x0.append(X[i]) 
    elif(D[i]==1):
        x1.append(X[i]) 
    elif(D[i]==2):
        x2.append(X[i]) 
x0=np.asarray(x0)
x1=np.asarray(x1)
x2=np.asarray(x2)
plt.scatter(x0[:, 0], x0[:, 1], c = "red", marker='o', label='label0')
plt.scatter(x1[:, 0], x1[:, 1], c = "green", marker='*', label='label1')
plt.scatter(x2[:, 0], x2[:, 1], c = "blue", marker='+', label='label2')
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend(loc=2)
plt.show()
