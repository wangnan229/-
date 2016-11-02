# coding=utf8
import numpy as np
import re

def loadDataSet():
    postingList=[['my','dog','has','flea','problems','help','please'],
                 ['maybe','not','take','him','to','dog','park','stupid'],
                 ['my','dalmation','is','so','cute','I','love','him'],
                 ['stop','posting','stupid','worthless','garbage'],
                 ['mr','licks','ate','my','steak','how','to','stop','him'],
                 ['quit','buying','worthless','dog','food','stupid']
                ]
    classVec=[0,1,0,1,0,1]  #1代表侮辱性文字  0代表非侮辱性文字
    return postingList,classVec


# arrayList=np.array(postingList)
# print type(arrayList),arrayList.shape

def createVocabList(dataSet): # set 也是用[]表示，set[]中不能再包含[]。
    vocabSet=set([])
    for doc in dataSet:
        vocabSet=vocabSet | set(doc)  # 取集合的并集操作
    # print 'Documents transformed to vocabulay set is:', vocabSet
    return list(vocabSet) # 需要把集合vocabSet转化为列表list

# 利用词汇表向量，将输入词向量转换为数字向量
def setOfWords2Vec(vocabList,inputSet): # vocabList,inputSet都各自是一个list
    returnVec=[0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]=1
        else:
            print '%s is not exist.' % word
    return returnVec

# 文档词袋模型
def bagOfWords2Vec(vocabList,inputSet):
    returnVec=[0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1 # 此处有不同
        else:
            # print '%s is not exist.' % word
            continue
    return returnVec

# 训练贝叶斯分类器
def classifyP(trainMat,classVec):
    trainMat=np.array(trainMat)
    numWords=trainMat.shape[1]
    numTrainDocs=trainMat.shape[0]
    p1Num=np.zeros(numWords)  # p1分类的矩阵初始化，用0填充一维的向量，书中用了ones()
    p0Num=np.zeros(numWords)
    p1Denom=0.0   # 书中用了2.0
    p0Denom=0.0
    for i in range(numTrainDocs):
        if classVec[i] == 1:
            p1Num += trainMat[i]  # 矩阵运算
            p1Denom += sum(trainMat[i])
        else:
            p0Num += trainMat[i]
            p0Denom += sum(trainMat[i])
    print 'p1Num is:',p1Num
    print 'p1Denom is:',p1Denom
    p1Vec=p1Num/p1Denom  # 侮辱性文档中各个词汇占所有侮辱性文档的总词汇数量的占比
    p0Vec=p0Num/p0Denom  # 非侮辱性文档中各个词汇占所有非侮辱性文档的总词汇数量的占比
    pAbusive=sum(classVec)/float(numTrainDocs)
    # print p1Vec,sum(p1Vec)
    # print p0Vec,sum(p0Vec)
    # print pAbusive
    return p1Vec,p0Vec,pAbusive


# 此处没有用书中的取自然对数的做法

def classifyNB(voca2Vec,p1Vec,p0Vec,pClass1):  # 对于双分类，一个分类的概率是pClass1，另一个分类的概率就是1-pClass1.如果是多分类，就不是这样。
    p1=sum(p1Vec*voca2Vec)+pClass1
    p0=sum(p0Vec*voca2Vec)+(1-pClass1)
    if p1>p0:
        # print 'p1 is %f,' % p1,'p0 is %f' % p0
        # print 'Classified as:',1
        return 1
    else:
        # print 'p1 is %f,' % p1, 'p0 is %f' % p0
        # print 'Classified as:',0
        return 0

def file2Str(file):
    with open(file) as f:
        s=''.join(f.readlines())
    s0=re.compile('\W*').split(s)
    # print s0
    s1=[tok.lower() for tok in s0 if len(tok)>2] # 去掉空格，并都转换为小写
    # print s1
    return s1

