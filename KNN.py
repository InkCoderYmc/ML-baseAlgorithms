'''
（1）计算已知类别数据集中的点与当前点之间的距离
（2）按照距离递增次序排序
（3）选取与当前点距离最小的k个点
（4）确定前k个点所在类别的出现频率
（5）返回前k个点出现频率最高的类别作为当前点的预测分类
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import operator

#KNN函数，输入已有数据，带预测数据，输出预测结果
def KNN(X_train, Y_train, X_test, k):
    m, n = X_train.shape
    #计算距离，n个特征值均求差值，再平方相加，即为距离的平方
    diffMat = np.tile(X_test,(m,1)) - X_train
    sq_diffMat = diffMat ** 2
    sq_distance = sq_diffMat.sum(axis = 1)
    distance = sq_distance ** 0.5 #开方，得到距离值
    sort_Index = distance.argsort() #将距离结果从小到大排序，提取索引值
    classCount = {}
    for i in range(k):
        voteIlabel = Y_train[sort_Index[i]]#得到k个距离最近的标签
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1#将得到的标签转化成字典形式，并设置value，key相同加1，key不同value为1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


#Load iris file
data=datasets.load_iris()
X=data['data']
y=data['target']

train_x, test_x, train_y, test_y = train_test_split(X, y, test_size = 0.5, random_state=33)
k = 5
y_prob = []
for i in range(test_x.shape[0]):
    x_input = test_x[i,:]
    knn_out = KNN(train_x, train_y, x_input, k)
    y_prob.append(knn_out)
count = 0
for i in range(np.shape(test_y)[0]):
    if y_prob[i] == test_y[i]:
        count += 1
accuracy = count / len(test_y)
