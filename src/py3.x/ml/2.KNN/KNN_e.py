from numpy import *
import operator
import os
from collections import Counter
import sys


def classfy0(inX, dataSet, labels, k):
    oriDistance = (dataSet - tile(inX, (dataSet.shape[0], 1))) ** 2
    distanceData = oriDistance.sum(axis=1) ** 0.5
    sortDistanceIndex = distanceData.argsort()

    lablesCount = {}
    for i in range(k):
        label = labels[sortDistanceIndex[i]]
        lablesCount[label] = lablesCount.get(label, 0) + 1

    lableCount = sorted(lablesCount.items(),
                        key=operator.itemgetter(1), reverse=True)
    return lableCount[0][0]


def classfy1(inX, dataSet, labels, k):
    distance = sum((dataSet - inX) ** 2, axis=1) ** 0.5
    print(distance)
    selectedLable = [labels[index] for index in distance.argsort()[0:k]]
    print(selectedLable)
    return Counter(selectedLable).most_common(1)[0][0]


def fileToMatrix(fileName):
    file = open(fileName, 'r')
    lines = len(file.readlines())
    rMat = zeros((lines, 3))
    rLabels = []
    index = 0
    file = open(fileName, 'r')
    for line in file.readlines():
        line = line.strip()
        data = line.split('\t')
        rMat[index] = data[0:3]
        rLabels.append(int(data[-1]))
        index += 1
    return rMat, rLabels


def autoNormal0(dataSet):
    mins = dataSet.min(0)
    maxs = dataSet.max(0)
    ranges = maxs - mins
    rows = dataSet.shape[0]
    norData = dataSet - tile(mins, (rows, 1))
    norData = norData / tile(ranges, (rows, 1))
    return norData


def autoNormal1(dataSet):
    mins = dataSet.min(0)
    maxs = dataSet.max(0)
    norData = (dataSet - mins) / (maxs - mins)
    return norData


def Test():
    dataSet = array([[1.0, 1.1], [8.9, 1.2], [0.1, 0.2], [0.2, 0.1]])
    labels = ['A', 'A', 'C', 'C']

    test = [0.1, 0.1]
    print("result %s " % classfy1(test, dataSet, labels, 3))


def TestData():
    tRadio = 0.1
    mat, labels = fileToMatrix("../../../../data/2.KNN/datingTestSet2.txt")
    normMat = autoNormal1(mat)
    rows = mat.shape[0]
    tRows = int(rows * tRadio)
    error = 0
    for i in range(tRows):
        outClass = classfy1(normMat[i], normMat[tRows:rows], labels[tRows:rows], 3)
        error += outClass != labels[i]

    print("error num: %s, percent %s" % (error, error / tRows))


def imgToMat(fileName):
    fr = open(fileName, 'r')
    rMat = zeros((1, 1024))
    for i in range(32):
        line = fr.readline()
        for j in range(32):
            rMat[0, 32 * i + j] = int(line[j])
    return rMat


def handWritingTest():
    trainingLabels = []
    trainingFileList = os.listdir("../../../../data/2.KNN/trainingDigits")
    trainingLen = len(trainingFileList)
    trainMat = zeros((trainingLen, 1024))
    for i in range(trainingLen):
        trainingFileName = trainingFileList[i]
        trainingLabels.append(int(trainingFileName.split(".")[0].split('_')[0]))
        trainMat[i] = imgToMat('../../../../data/2.KNN/trainingDigits/%s' % trainingFileName)

    testFileList = os.listdir("../../../../data/2.KNN/testDigits")
    testLen = len(testFileList)
    error = 0
    for i in range(testLen):
        testFileName = testFileList[i]
        testOut = int(testFileName.split(".")[0].split("_")[0])
        testMat = imgToMat("../../../../data/2.KNN/testDigits/%s" % testFileName)

        compueClass = classfy0(testMat, trainMat, trainingLabels, 3)
        error += testOut != compueClass
    print("error num: %s, percent: %s" % (error, error / testLen))


if __name__ == "__main__":

    # handWritingTest()
    Test()