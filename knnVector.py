# coding=utf8

from knn import knnClassify1,knnClassify2
import numpy as np
import os


def file2Vector(file):
    with open(file) as f:
        res=f.readlines()
        vector=np.zeros([1,1024])
        n=len(res)
        # print n
        for i in range(n):
            item=res[i].strip()
            # print item
            for j in range(32):
                vector[0,i*32+j]=item[j]
                # print vector
        # print vector
        return vector


# file='h:/9_71.txt'
# vector=file2Vector(file)
# print 'test vector is:',vector,vector.shape


# 构造Dataset、Labels
def BuildDataSet(dir):
    files=os.listdir(dir)
    # print files
    n=len(files)
    DataSet=np.zeros([n,1024])
    Labels=[]
    for i in range(n):
        vector=file2Vector('H:/trainingDigits/'+files[i])
        DataSet[i,]=vector
        Labels.append(int(files[i][0]))

    # print 'DataSet is:',DataSet,DataSet.shape
    # print labels,len(labels)
    return DataSet,Labels
# 经测试，knnClassify2的性能是knnClassify1的性能的一百倍，所以矩阵性能相比更高很多。。。。
# 将测试集的数据测试一遍，输出错误率，testdir跟traniningdir都放在了h盘，测试k为3时错误率低
testdir=os.listdir('h:/testDigits')
trainingdir='h:/trainingdigits'
trainingDataSet, trainingLabels = BuildDataSet(trainingdir)
n=0
for testfile in testdir:
    vector= file2Vector('h:/testDigits/'+testfile)
    testNum = knnClassify2(vector,trainingDataSet, trainingLabels, 3)
    trueNum = int(testfile[0])
    if testNum!=trueNum:
        n+=1.0
print 'Numbers of wrong classify is :%d,error ratio is:%f'%\
      (n,n/float(len(testdir)))
