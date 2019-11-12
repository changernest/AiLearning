#!/usr/bin/python
# -*- coding: UTF-8 -*-

import operator
from math import log
import decisionTreePlot as dtPlot
from collections import Counter

'''
决策树分类：已有带标记的数据集，依次计算按每个特征进行分类的信息熵，选出最优分类特征。
而后依据该特征对数据集分类，依次对每个子集循环计算，直至到具体分类标记，形成完整的决策树。
对新的测试数据，传入决策树，根据决策树依次决策，直至最终确定所属分类标记。
'''


def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    laels = ['no surfacing', 'flippers']
    return dataSet, laels


# for a dataSet, calculate different labels' entropy
def calcShannonEnt(dataSet):
    lableMap = {}
    totalNum = len(dataSet)
    for data in dataSet:
        label = data[-1]
        lableMap[label] = lableMap.get(label, 0) + 1
    shannonEnt = 0
    print("lableMap: ", lableMap)
    for key in lableMap:
        print("key is %s, lableMap[key] is %s" % (key, lableMap[key]))
        percent = float(lableMap[key]) / totalNum
        shannonEnt -= percent * log(percent, 2)
    return shannonEnt


def splitDataSet(dataSet, index, key):
    labelSet = []
    for data in dataSet:
        if data[index] == key:
            set = data[:index]
            set.extend(data[index + 1:])
            labelSet.append(set)
    return labelSet


def chooseBestShannonEnt(dataSet):
    baseShanonEnt = calcShannonEnt(dataSet)
    numFeatures = len(dataSet[0]) - 1
    bestGain, bestFeature = 0, 0
    for i in range(numFeatures):
        iData = [data[i] for data in dataSet]
        iSet = set(iData)
        newShannonEnt = 0
        for j in iSet:
            jData = splitDataSet(dataSet, i, j)
            jPerc = float(len(jData)) / len(dataSet)
            # for each iFeature, calculate each labels' entropy first,
            # then join in distribution to calculate total entropy.
            newShannonEnt += jPerc * calcShannonEnt(jData)
        # entropy is more smaller more better. so gain is more bigger more better.
        newGain = baseShanonEnt - newShannonEnt
        print('baseShanonEnt:', baseShanonEnt, 'newShannonEnt:', newShannonEnt, 'i=', i)
        if newGain > bestGain:
            bestGain = newGain
            bestFeature = i
    return bestFeature


def majorCount(classes):
    counter = {}
    for data in classes:
        counter[data] = counter.get(data, 0) + 1
    sortCounter = sorted(counter.items(), key=operator.getitem(1), reverse=True)
    return sortCounter[0][0]


def majorCountSimple(classes):
    return Counter(classes).most_common(1)[0][0]


# labels: dataSet's feature name list
def createTree(dataSet, featureNames):
    # classes: dataSet's tags.
    classes = [data[-1] for data in dataSet]
    if classes.count(classes[0]) == len(classes):
        return classes[0]
    if len(dataSet[0]) == 1:
        return majorCount(classes)

    # choose first best feature to make first decision step.
    iFeature = chooseBestShannonEnt(dataSet)
    # labels[iFeature]: best feature's name
    feature = featureNames[iFeature]
    print("delete:", iFeature, "labels:", featureNames)
    # labels should compare to dataSet's feature one by one.
    del featureNames[iFeature]
    tree = {feature: {}}
    iData = [data[iFeature] for data in dataSet]
    iSet = set(iData)
    # for each feature, iterate it's each value, and create a small tree choice.
    # Also, each small tree's bulid  will delete it's feature.
    # So subLabels is NEED,for labels will be empty when a small tree is created.
    for i in iSet:
        subFeatureNames = featureNames[:]
        tree[feature][i] = createTree(splitDataSet(dataSet, iFeature, i), subFeatureNames)
    print("tree:", tree)
    return tree


def classify(tree, featureNames, test):
    firstFeatureName = list(tree.keys())[0]
    subTree = tree[firstFeatureName][test[featureNames.index(firstFeatureName)]]
    if isinstance(subTree, dict):
        classLabel = classify(subTree, featureNames, test)
    else:
        classLabel = subTree
    return classLabel


def storeTree(tree, filename):
    import pickle
    with open(filename, 'wb') as fw:
        pickle.dump(tree, fw)


def grabTree(filename):
    import pickle
    with open(filename, 'rb') as fr:
        return pickle.load(fr)


def fishTest():
    dataSet, labels = createDataSet()
    import copy
    tree = createTree(dataSet, copy.deepcopy(labels))
    print("tree:", tree)
    print(classify(tree, labels, [0, 0]))
    dtPlot.createPlot(tree)


def lenseTree():
    fr = open('../../../../data/3.DecisionTree/lenses.txt')
    dataSet = [data.strip().split('\t') for data in fr.readlines()]
    features = ['age', 'prescript', 'astigmatic', 'tearRate']
    tree = createTree(dataSet, features)
    print("lense tree:", tree)
    dtPlot.createPlot(tree)


if __name__ == '__main__':
    fishTest()
    # lenseTree()
