'''
类别K的Softmax分数：
    s_k(x) = θT_k*x

归一化处理
    p_k = e^(s_k(x))/∑e^(s_j(x))


'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing


#
def computer_scores(x, weights):
    return np.dot(x, weights.T)
#定义softmax函数
def softmax_function(z):
    z = z - np.max(z)
    return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)

#交叉熵函数
def cross_entropy(m, y_, p_):
    loss = -(1 / m) * np.sum(y_ * np.log(p_))
    return loss

#softmax_regression函数
def softmax_regression(X, Y, k, alpha, iters):
    m,n = X.shape
    weights = np.random.rand(k, n)
    all_losses = []
    for i in range(iters):
        scores = computer_scores(X, weights)
        p_ = softmax_function(scores)
        y_predict = np.argmax(p_, axis = 1)[:, np.newaxis]
        y_one_hot = np.zeros((m, k))
        y_one_hot[np.arange(m), Y.T] = 1
        loss = cross_entropy(m, y_one_hot, p_)
        all_losses.append(loss)
        deltaJ = (1 / m) * np.dot(X.T, (p_ - y_one_hot))
        weights = weights - alpha * deltaJ.T
    return weights, all_losses

#预测函数
def predict_prob(X, weights):
    scores = computer_scores(X, weights)
    p_ = softmax_function(scores)
    return np.argmax(p_, axis=1)[:, np.newaxis]

if __name__ == '__main__':
    #根据数据获取k的值
    data = datasets.load_iris()
    X = data['data']
    Y = data['target']
    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size = 0.5, random_state=33)
    k = len(np.unique(Y))
    weights, all_losses = softmax_regression(train_x, train_y, k, alpha=0.01, iters=800)
    p_out = predict_prob(test_x, weights)
    count = 0
    for i in range(np.shape(test_y)[0]):
          if p_out[i,] == test_y[i,]:
               count += 1
    accuracy = count / len(test_y)
