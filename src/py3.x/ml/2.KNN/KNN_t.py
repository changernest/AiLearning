from collections import Counter

from numpy import *
import os
import operator

'''
KNN分类：
已有一个带标记的数据集；将测试数据跟现有数据集做距离计算，
选出距离最近的K个数据，采用多数表决确定最终分类。
'''


def createDataSet():
    data = array([[0, 0], [1, 1], [0.2, 0.2], [0.9, 0.9]])
    label = array(['B', 'A', 'B', 'A'])
    return data, label


def classfyStd(inX, data, label, k):
    rows = data.shape[0]
    distanceMat = data - tile(inX, (rows, 1))
    sqrMat = distanceMat ** 2
    sumMat = sqrMat.sum(axis=1)
    proMat = sumMat ** 0.5
    indexes = proMat.argsort()
    count = {}
    for i in range(k):
        elabel = label[indexes[i]]
        count[elabel] = count.get(elabel, 0) + 1
    # count.items(): return tuple list by key and value
    # sorted(list, key=operator.itemgetter(1), reverse=True): sort list by each tuple's second element.
    return sorted(count.items(), key=operator.itemgetter(1), reverse=True)[0][0]


def classfySimple(inX, data, label, k):
    # .argsort(): return element's index list by element's desc sort
    indexes = (sum(((inX - data) ** 2), axis=1) ** 0.5).argsort()
    klebels = [label[index] for index in indexes[0:k]]
    # Counter(list).most_comment(1): return tuple list by element and it's occurrence number
    # most_common(1): [(1, 3)] ; most_common(3):[(3, 1), (2, 1), (1, 1)]
    result = Counter(klebels).most_common(3)[0][0]
    return result


def test1():
    data, lebal = createDataSet()
    print(classfySimple([0, 0.1], data, lebal, 2))
    print(classfyStd([0, 0.1], data, lebal, 2))


def readFile(filename):
    fr = open(filename, 'r')
    rows = len(fr.readlines())
    data = zeros((rows, 3))
    label = []
    fr = open(filename, 'r')
    index = 0
    for line in fr.readlines():
        line = line.strip()
        row = line.split('\t')
        data[index] = row[0:3]
        label.append(int(row[-1]))
        index += 1
    return data, label


def normDataStd(data):
    # tile(arr,(x, y)): copy arr's row x times, copy arr's line y times.
    return (data - tile(data.min(0), (len(data), 1))) / tile(data.max(0) - data.min(0), (len(data), 1))


def normDataSimple(data):
    # numpy broadcast: different shape's array or matrix can convert to each other for computation.
    return (data - data.min(0)) / (data.max(0) - data.min(0))


def testFile():
    data, label = readFile("../../../../data/2.KNN/datingTestSet2.txt")
    data = normDataStd(data)
    m = data.shape[0]
    testCount = int(m * 0.1)

    errorCount = 0
    for i in range(testCount):
        result = classfySimple(data[i], data[testCount:m], label[testCount:m], 3)
        # bool true 1; false 0
        errorCount += label[i] != result
    print("error percent %f , count %d" % ((errorCount / testCount), errorCount))


def imgToVector(filename):
    fr = open(filename, 'r')
    arr = zeros((1, 1024))
    for i in range(32):
        line = fr.readline()
        for j in range(32):
            arr[0, i * 32 + j] = int(line[j])
    return arr


# "../../../../data/2.KNN/trainingDigits" testDigits
def handWriteTest():
    files = os.listdir("../../../../data/2.KNN/trainingDigits")
    labels = []
    alen = len(files)
    trainSet = zeros((alen, 1024))
    for i in range(alen):
        labels.append(int(files[i].split(".")[0].split("_")[0]))
        trainSet[i] = imgToVector("../../../../data/2.KNN/trainingDigits/%s" % files[i])

    tfiles = os.listdir("../../../../data/2.KNN/testDigits")
    tlen = len(tfiles)
    errCount = 0
    for i in range(tlen):
        tlabel = int(tfiles[i].split(".")[0].split("_")[0])
        tSet = imgToVector("../../../../data/2.KNN/testDigits/%s" % tfiles[i])
        result = classfyStd(tSet, trainSet, labels, 3)
        errCount += tlabel != result
    print("error precent %f, error count is %d " % (errCount / tlen, errCount))


if __name__ == '__main__':
    # test1()
    # testFile()
    handWriteTest()
