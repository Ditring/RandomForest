#encoding=utf8
import json
import random
from math import log
from DecideByRandomForest import voteType





def classifyDataSet(dataSet, axis, value):  # 获取按某个属性的某一值分类后的集合,剔除某一属性
    reDataSet = []
    for featureVector in dataSet:
        if featureVector[axis] == value:
            reducedFeatureVector = featureVector[:axis]
            reducedFeatureVector.extend(featureVector[axis + 1:])
            reDataSet.append(reducedFeatureVector)
    return reDataSet


def chooseBestFeatureToSplit(dataSet):  # 选择最优的分类属性
    infoEnt = calculateInfoEnt(dataSet)  # 原始的熵
    bestInfoGain = 0
    bestFeature = -1
    for i in range(len(dataSet[0]) - 1):
        featureList = [example[i] for example in dataSet]
        uniqueVals = set(featureList)
        conditionalEnt = 0
        # splitInfo = 0.0
        for value in uniqueVals:  # 遍历所有该属性的不同取值
            subDataSet = classifyDataSet(dataSet, i, value)  # 获取该属性取值为value的集合
            conditionalEnt += (float(len(subDataSet)) / len(dataSet)) * calculateInfoEnt(subDataSet)  # 按属性值分类后的熵
        infoGain = infoEnt - conditionalEnt  # 计算信息增益，跳出循环的时候，infoGain即为该属性的信息增益
        if (infoGain >= bestInfoGain):  # 选取信息增益最大的属性#注意这里要是大于等于，不能是大于，是大于的话，当只剩两个，且其中一个比较纯的时候，就会出现不能对bestFeature赋值的情况，那时返回bestFeature的是-1，就会导致出现bug
            #并且这种情况只在这一个属性纯的情况下发生
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature




def createTree(dataSet, labels):  # 递归建树
    typeList = [example[-1] for example in dataSet]  # 将类别列赋值给typeList
    if typeList.count(typeList[0]) == len(typeList):  # 当该集合已经纯了，则返回类别
        return typeList[0]
    if len(dataSet[0]) == 1:  # 属性用的只剩一个了，那么根据个数来判断好坏
        return voteType(typeList)
    bestFeature = chooseBestFeatureToSplit(dataSet)  # 选择决策节点
    bestFeatureLabel = labels[bestFeature]
    myTree = {bestFeatureLabel: {}}  # 分类结果以字典形式保存
    del (labels[bestFeature])
    featureValues = [example[bestFeature] for example in dataSet]
    uniqueVals = set(featureValues)
    for value in uniqueVals:
        subLabels = labels[:]  # 复制一个subLabels供使用
        myTree[bestFeatureLabel][value] = createTree(classifyDataSet(dataSet, bestFeature, value), subLabels)
    return myTree

def calculateInfoEnt(dataSet):  # 计算数据的信息熵
    labelCounts = {}
    for featureVector in dataSet:
        currentLabel = featureVector[-1]  # 每行数据的最后一个属性（类别）（好瓜坏瓜）
        if currentLabel not in labelCounts.keys():  # 如果该类别没有加入过，则加入到labelCounts中,这么写的好处是，你无法判断数据集的类别有哪些，不需要再进行传入了
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1  # 统计有多少个类以及每个类的数量
    infoEnt = 0
    for type in labelCounts:
        infoEnt -= (float(labelCounts[type]) / len(dataSet)) * log(float(labelCounts[type]) / len(dataSet), 2)  # 求和获取信息熵
    return infoEnt


def treeVote(tree,object):
    if '好瓜' == tree or '坏瓜' == tree:
        return tree
    for feature in tree.keys():
        for key in tree.get(feature):
    #用增强for对python字典的get方法获取某一key的value，对get的result进行遍历，如果该result还是个字典的话，
    # 对get的result进行遍历是对result的key进行遍历
    #噢，python中对字典用增强for遍历，获取的是其内容key
            if key in object:
                return treeVote(tree.get(feature).get(key), object)

# def voteType(typeList):  # 在叶子节点处（属性节点用完了），如果还是多种类别混杂，则根据哪种类别最多，就判为哪种类别，同理此方法也可以用于最后森林投票处，把所有投票结果放进来，然后获取最大的
#     typeCount = {}
#     for vote in typeList:
#         if vote not in typeCount.keys():
#             typeCount[vote] = 0
#         typeCount[vote] += 1
#     sortedTypeCount = sorted(typeCount.items(), key=operator.itemgetter(1), reverse=True)
#     return sortedTypeCount[0][0]



if __name__ == '__main__':
    treeSet=[]
    f=open("json/labels.json", "r")
    labels=json.loads(f.read())
    f=open("json/dataset.json", "r")
    dataSet=json.loads(f.read())
    for i in range(100):
        randomLabels = random.sample(labels[:len(labels) - 1], 3)
        randomLabels.append(labels[len(labels) - 1])
        newDataSet = []
        k = 0
        for vector in dataSet:
            tmp = [0]*len(randomLabels)
            for j in range(len(labels)):
                if labels[j] in randomLabels:
                    tmp[randomLabels.index(labels[j])] = vector[j]
            newDataSet.append(tmp)
        newDataSet=random.choices(newDataSet, k=len(newDataSet))
        treeSet.append(createTree(newDataSet, randomLabels))
    f=open("json/treeset.json", "w")
    f.write(json.dumps(treeSet))
    # object=input()
    # typeList=[]
    # for tree in treeSet:
    #     typeList.append(treeVote(tree,object))
    # print(voteType(typeList))
