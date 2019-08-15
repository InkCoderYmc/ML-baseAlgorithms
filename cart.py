import numpy as np
from collections import Counter, defaultdict
import math
from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
import operator

def Gini(y_data):
    m = len(y_data)
    count = {}
    num = 0.0
    for i in y_data:
        if i in count:
            count[i] += 1
        else:
            count[i] = 1
    for item in count.keys():
        num = num + pow(1.0 * count[item] / m,2)
    return (1.0-num)

#属性a的基尼指数
def  Gain_Gini(D_data, D_target, list_num):
    #v表示属性a的v个可能的取值
    a = list(D_data[:,list_num])
    D_target = list(D_target[:])
    N = len(a)
    #根据属性a的数值对D_data进行分类
    D_ = {}
    Gini_D_a = {}
    for i in range(N):
        if a[i] not in D_:
            D_[a[i]] = []
        D_[a[i]].append(D_target[i])
    for item in D_.keys():
        n1 = len(D_[item])#当前分割点的n1
        temp = D_target[:]
        #删除D_[item]中对应的数据
        for i in D_[item]:
            temp.remove(i)
        n2 = len(temp)#当前分割点的n2
        #到此得到分割后的两部分数据集D_[item]、temp，分割长度n1、n2
        #计算当前分割点的Gain_GINI
        G_GINI = n1 / N * Gini(D_[item])+ n2 / N * Gini(temp)
        Gini_D_a[item] = G_GINI
        #将计算得到的Gain_GINI进行排序，输出最小值
        out = sorted(Gini_D_a.items(), key=operator.itemgetter(1))
    return out[0]
        
#计算所有特征，得到最小的基尼指数和对应的特征
def choose_feature(x_data,y_data):
    w = np.size(x_data,axis=1)
    count = []
    count_label = {}
    for i in range(w):
        a = Gain_Gini(x_data,y_data,i)
        count.append(a[1])
        count_label[i] = a
    id = count.index(min(count))
    return id,count_label[id][0]   

#A为空时，获取样本数最多的类
def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

#构建决策树
'''
决策树学习基本算法：
输入：训练集D = {(x1, y1), (x2, y2), ……,(xm, ym)};
            属性集A = {a1, a2, ……， ad}
过程：函数TreeGenerate(D, A)
生成结点node;
if D中样本全属于同一类别C then
    将node标记为C类叶节点；
    return;
end if 
if A=∅ or D中样本在A上取值相同 then
    将node标记为叶结点， 其类别标记为D中样本数最多的类;
    return;
end if
从A中选择最优划分属性a_*;——使用基尼指数
for a_*的每一个值a__v_* do:
    为node生成一个分支;令D_v表示D中在a_*上取值为a__v_*的样本子集;
    if D_v为空 then
        将分支结点标记为叶结点，其类别标记为D中样本最多的类;
        return
    else
        以TreeGenerate(D_v, A\{a_*})为分支结点
    end if
end for
输出：以node为根节点的一颗决策树
'''

def TreeGenerate(dataSet, target, feature):
    
    #如果数据集中的所有数据都属于同一类，将其标记被该类别，并返回
    if len(set(target))==1:
        return target[0]
    
    #若特征集为空，将数据集中现存最大的类作为该节点的标记
    if len(dataSet[0])==1:
        labelCount = {}
        labelCount = dict(Counter(target))
        return max(labelCount, key=labelCount.get)
    
    # 获取最佳划分属性
    bestFeature, gini_value = choose_feature(dataSet, target)
    bestFeatureLable = feature[bestFeature]
    #构建树，以所得特征为子结点
    decisionTree = {bestFeatureLable:{}}
    #删除已使用的特征
    del feature[bestFeature]
    
    #获取a_v
    a_v = set(dataSet[:,bestFeature])
    for a_x in a_v:
        subFeature = feature[:]
        #获取对应的D_v
        D_v = dataSet[list(dataSet[:, bestFeature]==a_x)]
        D_v_target = target[list(dataSet[:, bestFeature]==a_x)]            
        decisionTree[bestFeatureLable][a_x] = TreeGenerate(D_v, D_v_target, subFeature)
            
    return decisionTree



#数据的加载及划分
data=datasets.load_iris()
X=data['data']
y=data['target']
feature=data['feature_names']
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size = 0.2, random_state=33)

tree = TreeGenerate(X, y, feature)
print(tree)
