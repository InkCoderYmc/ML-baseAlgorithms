import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split

#将数据集进行分类
def separateByClass(dataset, target):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i] #假设最后一个值为类别值
        if (target[i] not in separated):
            separated[target[i]] = []
        separated[target[i]].append(vector)
    return separated

#计算均值
def mean(numbers):
    return sum(numbers)/float(len(numbers))

#计算标准差
def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
    return math.sqrt(variance)

#对全部数据集提取特征
def summarize(dataset):
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    return summaries

#对分类提取特征
def summarizeByClass(dataset,target):
    separated = separateByClass(dataset, target)
    summaries = {}
    for classValue, instances in separated.items():
        summaries[classValue] = summarize(instances)
    return summaries

#计算正态分布的概率密度函数
def calculateProbability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

#计算概率，返回概率最大的标签值
def calculateClassProbabilities(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] *= calculateProbability(x, mean, stdev)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel

#预测函数
def prodect(summaries, test_x, test_y):
    y_prob = []
    for i in range(test_x.shape[0]):
        x_input = test_x[i,:]
        prob_out = calculateClassProbabilities(summaries, x_input)
        y_prob.append(prob_out)
    count = 0
    for i in range(np.shape(test_y)[0]):
        if y_prob[i] == test_y[i]:
            count += 1
    accuracy = count / len(test_y)
    return accuracy, y_prob

data=datasets.load_iris()
X=data['data']
y=data['target']
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size = 0.2, random_state=33)
summaries = summarizeByClass(train_x, train_y)
accuracy, y_prob = prodect(summaries, test_x, test_y)
print(accuracy, y_prob)