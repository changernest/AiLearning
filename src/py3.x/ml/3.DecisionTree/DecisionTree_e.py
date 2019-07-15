#!/usr/bin/python
# -*- coding: UTF-8 -*-

import operator
from math import log
import decisionTreePlot as dtPlot
from collections import Counter


def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    laels = ['no surfacing', 'flippers']
    return dataSet, laels


def calcShannonEnt(dataSet):
    lableMap = {}
    totalNum = len(dataSet)
    for data in dataSet:
        label = data[-1]
        lableMap[label] = lableMap.get(label, 0) + 1
    shannonEnt = 0
    for key in lableMap:
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
    bestGain, bestFeature = 0, 0;
    for i in range(numFeatures):
        iData = [data[i] for data in dataSet]
        iSet = set(iData)
        newShannonEnt = 0
        for j in iSet:
            jData = splitDataSet(dataSet, i, j)
            jPerc = float(len(jData)) / len(dataSet)
            newShannonEnt += jPerc * calcShannonEnt(jData)
        newGain = baseShanonEnt - newShannonEnt
        print('baseShanonEnt:', baseShanonEnt, 'newShannonEnt:', newShannonEnt, 'i=', i)
        if newGain > bestGain:
            bestGain = newGain
            bestFeature = i
    return bestFeature;


def majorCount(classes):
    counter = {}
    for data in classes:
        counter[data] = counter.get(data, 0) + 1
    sortCounter = sorted(counter.items(), key=operator.getitem(1), reverse=True)
    return sortCounter[0][0]


def createTree(dataSet, labels):
    classes = [data[-1] for data in dataSet]
    if classes.count(classes[0]) == len(classes):
        return classes[0]
    if len(dataSet[0]) == 1:
        return majorCount(classes)

    iFeature = chooseBestShannonEnt(dataSet)
    # print("iFeature:", iFeature, "labels:", labels)
    feature = labels[iFeature]
    print("delete:", iFeature,"labels:",labels)
    del labels[iFeature]
    tree = {feature: {}}
    iData = [data[iFeature] for data in dataSet]
    iSet = set(iData)
    # for each feature, iterate it's each value, and create a small tree choice.
    # Also, ach small tree's bulid  will delete it's feature.
    # So subLabels is NEED,for labels will be empty when a small tree is created.
    for i in iSet:
        subLabels = labels[:]
        tree[feature][i] = createTree(splitDataSet(dataSet, iFeature, i), subLabels)
    print("tree:", tree)
    return tree


def classify(tree, labels, test):
    first = list(tree.keys())[0]
    second = tree[first][test[labels.index(first)]]
    if isinstance(second, dict):
        classLabel = classify(second, labels, test)
    else:
        classLabel = second
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
    # fishTest()
    lenseTree()
