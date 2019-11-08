import numpy as np


def dataSet():
    trainData = [
        ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'gar e'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]
    return trainData, classVec


def wordSet(dataList):
    wordSet = set()
    for data in dataList:
        wordSet = wordSet | set(data)
    return list(wordSet)


def dataVec(wordSet, inData):
    dataVec = [0] * len(wordSet)
    for w in inData:
        if w in wordSet:
            dataVec[wordSet.index(w)] = 1
    return dataVec


def trainP(trainM, classVec):
    pc1 = np.sum(classVec) / len(trainM)

    p1Num = np.ones(len(trainM[0]))
    p0Num = np.ones(len(trainM[0]))

    p1NumAll = 2
    p0NumAll = 2

    for i in range(len(trainM)):
        if classVec[i] == 1:
            p1Num += trainM[i]
            p1NumAll += np.sum(trainM[i])
        else:
            p0Num += trainM[i]
            p0NumAll += np.sum(trainM[i])

    p1Vec = np.log(p1Num / p1NumAll)
    p0Vec = np.log(p0Num / p0NumAll)

    return p1Vec, p0Vec, pc1


def bayes(inVec, p1Vec, p0Vec, pc1):
    tPc1 = np.sum(inVec * p1Vec) + np.log(pc1)
    tPc0 = np.sum(inVec * p0Vec) + np.log(1 - pc1)
    print("bayes", tPc1, tPc0)
    if tPc1 > tPc0:
        return 1
    else:
        return 0


def testBayes():
    trainData, classVec = dataSet()
    trainSet = wordSet(trainData)
    print("trainSet", trainSet)
    trainM = []
    for line in trainData:
        trainM.append(dataVec(trainSet, line))

    print("trainP", np.array(trainM), np.array(classVec))

    p1Vec, p0Vec, pc1 = trainP(np.array(trainM), np.array(classVec))
    print("trainP", p1Vec, p0Vec, pc1)

    test_one = ['love', 'my', 'dalmation']
    print('the result is: {}'.format(bayes(np.array(dataVec(trainSet, test_one)), p1Vec, p0Vec, pc1)))

    test_two = ['stupid', 'garbage']
    print('the result is: {}'.format(bayes(np.array(dataVec(trainSet, test_two)), p1Vec, p0Vec, pc1)))


if __name__ == "__main__":
    testBayes()
