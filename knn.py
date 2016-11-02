# coding=utf8
from numpy import *
import operator
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time



def createDataSet():
    group=array([[1.0,1.1],
                 [1.0,1.0],
                 [0,0],
                 [0,0.1]
    ])
    labels=['A','A','B','B']
    return group,labels


group,labels=createDataSet()
# print 'Array维度是',group.shape


"""
计算输入向量与训练样本集的距离
选取距离最小的k个值
计算k个值中各个标签类别出现的频率
选取频率最大的分类标签为预测分类
"""
# 自己写的原生代码
def knnClassify1(inX,dataset,labels,k): # inX是一维矩阵（一维向量），dataset是二维矩阵
    start=time.time()
    print 'knnClassify1 start'
    n=dataset.shape[0]
    disLabels=[]
    m=inX.shape[1]
    print 'inX.shape[1] is :',inX.shape[1]
    for i in range(n):
        res = {'dis': '', 'label': ''}
        distance=0
        for j in range(m):
            # print 'inX[0,m] is:',inX[0,j]
            d=(inX[0,j]-dataset[i,j])**2 #把轴上的差的平方进行相加
            distance+=d
        res['dis']=distance**0.5
        res['label']=labels[i]
        disLabels.append(res)
    sortedRes=sorted(disLabels,key=lambda item : item['dis'] , reverse=False)  # 用disLabels列表中的元素（字典）的属性（dis键）进行顺序排序
    # print sortedRes
    kRes=[]
    for i in range(0,k):
        kRes.append(sortedRes[i])

    classCount={}
    for i in kRes:
        labellist=i.values()
        label=labellist[0]
        classCount[label]=classCount.get(label,0)+1

    sortedClassCount=sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    # return kRes,classCount,sortedClassCount,sortedClassCount[0][0]
    print 'knnClassify1 over,Time elapsed:',time.time()-start
    return sortedClassCount[0][0]


# 机器学习实战的kNN代码，运用了矩阵的加减运算，跟字典的排序。
def knnClassify2(inX,dataset,labels,k):
    start = time.time()
    # print 'knnClassify2 start'
    datasetSize=dataset.shape[0]
    diffArray=tile(inX,(datasetSize,1))-dataset
    sqDistances=diffArray**2
    distances=(sqDistances.sum(axis=1))**0.5  # axis=1 如果不加这个参数，sum会把所有的和进行相加
    sortedDistIndicies=distances.argsort()

    classCount={}
    for i in range(k):  # range(k)是从[0,k-1]，基本点
        label=labels[sortedDistIndicies[i]]
        classCount[label]=classCount.get(label,0)+1
    # print classCount
    sortedClassCount=sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)   #iteritems()方法
    # return distances, sortedDistIndicies,sortedClassCount,sortedClassCount[0][0]  # 返回名次
    # print 'knnClassify2 over,Time elapsed:', time.time() - start
    return sortedClassCount[0][0]
'''
备注：
k=1:[distances0] sdi[3]
k=2:[distances0] sdi[3],[distances1] sdi[2] ,
k=3:[distances0] sdi[3],[distances1] sdi[2] ,[distances3] sdi[1]  将这个顺序倒过来就是。
'''
inX=array([[0.2,0.3]])
'''
classifyRes1=knnClassify1(inX,dataset=group,labels=labels,k=3)
classifyRes2=knnClassify2(inX,dataset=group,labels=labels,k=3)
print classifyRes1
print classifyRes2
'''

# 这个转化txt为Matrix的方法，应该会根据场景需求进行更改。比如其中的labels如果为非数字字符串，就需要去掉int等等。
def file2matrix(file):
    with open(file,'rb') as f:
        data=f.readlines()
        n=len(data)
        datamatrix=np.zeros((n,3))  # np.zeros((n,3))
        index=0
        labels=[]
        for row in data:
            row=row.strip()
            row=row.split('\t')
            datamatrix[index,:]=row[0:3]
            labels.append(int(row[-1]))
            index+=1
        return datamatrix,labels

# file='h:/datingtestset2.txt'
# dataDatingMat,datingLabels=file2matrix(file)
# print dataDatingMat
# print datingLabels

# fig=plt.figure()
# ax=fig.add_subplot(111)
# ax.scatter(x=dataDatingMat[:,0],y=dataDatingMat[:,1],  # matrix(:,1)表示矩阵的第2列玩视频游戏百分比，matrix(:,2)表示矩阵的第3列消耗冰淇淋公升数
#            s=20.0*array(datingLabels),c=20*array(datingLabels),marker='v')  # s表示size，c表示colors
# plt.xlabel('FlyCost')
# plt.ylabel('GameCost')
# plt.show()


# Z变换：对datingTestSet的三个字段的数据进行Z变换，即对每个字段，先求出最大\最小值，然后轮寻每个数值，根据公式求出变换后的值。

def zNorm(dataset):
    minValue=dataset.min(axis=0)   # 参数为0，生成的才是3个字段的最小值的一维矩阵
    maxValue=dataset.max(axis=0)   # 参数为0
    zNormDataSet=np.zeros(dataset.shape)
    m=dataDatingMat.shape[0]
    # tmpMinValue=tile(minValue,(m,1))
    # print tmpMinValue
    zNormDataSet=(dataset-tile(minValue,(m,1)))
    zNormDataSet=zNormDataSet/(tile(maxValue-minValue,(m,1)))
    return zNormDataSet,minValue,maxValue

# zNormDataSet,minValue,maxValue=zNorm(dataDatingMat)
# print zNormDataSet,minValue,maxValue


def datingClassTest(ratio,k):
    m=zNormDataSet.shape[0]
    testRange=int(m*ratio)
    errorCount=0
    for i in range(testRange):
        result=knnClassify1(inX=zNormDataSet[i,:],dataset=zNormDataSet[testRange:m,:],labels=datingLabels[testRange:m],k=k)
        print 'knnClassify2 result is :%d ,real value is :%d  ' % (result,datingLabels[i])
        if result != datingLabels[i]:
            errorCount+=1
    print 'errorCount is: %d' % errorCount
    errorRatio=errorCount/float(testRange)
    print 'errorRatio is: %f' % errorRatio

# datingClassTest(ratio=0.10,k=5)


# 最终：需要输入N个初始值：每个字段的初始值，这里包括飞行里程数、冰淇淋消耗数、打游戏时间占比。
def classifyPerson():
    flyMiles=float(raw_input('Please input fly miles:'))
    iceConsume=float(raw_input('Please input ice consume:'))
    gameTime=float(raw_input('Please input game time:'))
    inst=np.array([flyMiles,iceConsume,gameTime])
    dataDatingMat, datingLabels = file2matrix(file)
    zNormDataSet,minValue,maxValue=zNorm(dataDatingMat)

    res=knnClassify1(inX=(inst-minValue)/(maxValue-minValue),  # 矩阵加减法
                 dataset=zNormDataSet,
                 labels=datingLabels,
                 k=5)
    resultList=['didntLike','smallDoses','largeDoses']
    print res,resultList[res-1]


# classifyPerson()
