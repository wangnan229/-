# coding=utf8
import jieba
import jieba.analyse
import re
import numpy as np
import os
import math
import random
# 将已分类的文档做成矩阵、标签两个列表，为再将矩阵做成不重复的词向量。
# 文档向量化：将已分类的文档随机抽取一定比例，每篇文档以标准词向量转换为词袋模型的文档词向量，做成文档训练集。
# 训练分类器：对于文档训练集中的每一个文档向量，如果其分类为a，那么对应的Pa分类向量PaVec+1，同时Pa分类的总词数+上这个文档的总词数sum(PaVec)，
# Pa分类的词概率向量Pa=PaVec/sum(PaVec)。同样得出其他分类的Pn值。再计算Pn分类的文档在总文档数的概率。
# 应用分类器：待分类的文档向量TestVec，各个分类Pn的PnVec、Pn，选出最大值。
# 模型1：没有计算topK词汇的权重。下一步，准备模型2，加入topK权重。
#
# 将N个分类目录下的文档分别合并成指定分类名称的N个文档集，然后分别切词，输出一个矩阵，矩阵包含N个带有TF-IDF权重的词向量，每个词向量含topK个词组。


def file2Vec(file):
    a=[]
    with open(file,'rb') as f:
        cont0=f.readlines()
        for line in cont0:
            line=re.sub('\s','',line)
            a.append(line)
    fileStr =''.join(a)
    jieba.load_userdict('dict.txt')
    fileVec0 = jieba.analyse.extract_tags(fileStr, topK=10, withWeight=True,withFlag=False)
    fileVec=[]
    for voc in fileVec0:
        # print voc
        voc=list(voc)
        fileVec.append(voc)
    return fileVec

def file2cont(filesList):
    conts=''
    for file in filesList:
        with open(file, 'rb') as fr:
            cont = fr.read()
            conts += cont.decode('gbk')
    return conts


def article_labels(filedir,label,testRatio):   # 输出该分类文档库中test的向量集[[],[],...[]]跟该分类训练集的topK词汇。
    files=os.listdir(filedir)
    filesList=[]
    labelsList=[]
    for file in files:
        file = re.sub('[\r\n]', '', file.decode('gbk'))
        file=filedir+'/'+file
        filesList.append(file)

    testFiles=[]
    testLabels=[]

    numFiles=len(filesList)
    numTest=int(numFiles * testRatio)
    for i in range(numTest):
        a=random.randint(0, numFiles - 1 - i)
        # print a
        testFile = filesList[a]  # 随机挑选一个file，随后删除
        # print testFile
        testFiles.append(testFile)
        del filesList[a]
        testLabels.append(label)

    trainFiles=filesList
    trainCont=file2cont(trainFiles)
    trainVocList=[]
    jieba.load_userdict('dict.txt')
    trainVocList.append(jieba.analyse.extract_tags(trainCont, topK=10, withWeight=True,withFlag=False))

    return trainVocList,testFiles,testLabels

labelsClassify=[['tech','technology'],['finance','finance'],['baby','baby']] # file path,label

trainList=[]
trainLabels=[]
testFiles=[]
testLabels=[]

for label in labelsClassify:
    trainList1,testFiles1,testLabels1 = article_labels(filedir=label[0],label=label[1],testRatio=0.2)
    trainLabels.append(label[1])  # trainLabels不在article_labels函数中
    trainList.extend(trainList1)
    testFiles.extend(testFiles1)
    testLabels.extend(testLabels1)
print 'trainList is:',trainList
print 'trainLabels is:',trainLabels
print 'testFiles is:',testFiles
print 'testLabels is:',testLabels

vocStdSet=set([])
for class_ in trainList:
    for voc in class_:
        vocStdSet=set([voc[0]]) | vocStdSet
vocStdList=list(vocStdSet)  # 生成标准词向量
# print vocStd
print 'vocStdList is:'
for i in range(len(vocStdList)):
     print i+1,vocStdList[i]

weightPlus=[0]*len(vocStdList)
numVoc=[0]*len(vocStdList)
for class_cont in trainList:
    for voc in class_cont:
        if voc[0] in vocStdList:
            # numVoc[vocStdList.index(voc[0])] += 1  # 统计词频，同模型1不同，因为是所有文档合并分词，所以此处都是1
            weightPlus[vocStdList.index(voc[0])] += voc[1]  # 获取词TF-IDF值
weightsPlusMat=np.array(weightPlus)
numVocMat=np.array(numVoc)
# print 'weightsPlusMat is:',weightsPlusMat
weightsMat=weightsPlusMat # 同模型1不同，因为是所有文档合并分词，直接使用分词后的权重
print 'weightsMat is:',weightsMat

# 将单文档按照标准词向量生成文档词向量
def class2stdVec(vocStdList,classVec):
    numStdVec=len(vocStdList)
    # print  'len(vocStdList) is:',numStdVec
    class2StdVec=np.zeros(numStdVec)
    for voc in classVec:
        if voc[0] in vocStdList:    # 此处同模型1有所不同
            class2StdVec[vocStdList.index(voc[0])] += voc[1] # 把待分类文档的词组原始权重传给标准化后的词向量
    # print class2StdVec
    return list(class2StdVec)
# 将文档词向量集转换为文档标准数字向量集
classVecStd=[]
for class_cont in trainList:
    classVecStd.append(class2stdVec(vocStdList,class_cont))
print 'classVecStd is:',classVecStd
print len(classVecStd)


def NBclassifyTrain(vocStdList,classVecStd,labels,weightsMat):
    PnNum={'P_' + label:np.zeros(len(vocStdList)) for label in labels } # 初始化各分类的PnNum向量
    PnDenom={'P_' + label +'_denom':0.0 for label in labels }  # 初始化各分类的PnDenom值
    numClass=len(classVecStd)
    trainMat = np.array(classVecStd)


    # 保存训练集中各个分类的PnDenom值
    for label in labels:
        for i in range(numClass):
            if labels[i] == label:
                PnNum['P_' + label] += trainMat[i]
                PnDenom['P_' + label +'_denom'] += sum(trainMat[i])
            else:
                continue
    print 'PnNum is:',PnNum
    # print 'PnDenom is:', PnDenom
    PnVec={'P_'+label+'_Vec':PnNum['P_' + label]* pow(weightsMat,2) / PnDenom['P_' + label +'_denom']  for label in labels}
    print PnVec
    Pclass=1/float(numClass)  # 有几个分类，Pclass就是几分之一
    return PnDenom,PnVec,Pclass

Pndenom,PnVec,Pclass=NBclassifyTrain(vocStdList,classVecStd,trainLabels,weightsMat)

# 测试，将trainMat分为80%的训练集跟20%的测试集，对测试集中的向量逐个测试。

errors=0
for testFile in testFiles:
    testVec=file2Vec(testFile)
    testVecStd = np.array(class2stdVec(vocStdList, testVec))
    # print testVecStd
    res=[[label,sum(Pvec * testVecStd) + Pclass] for label, Pvec in PnVec.iteritems()]
    # print res
    res_sorted=sorted(res,key=lambda item:item[1],reverse=True)
    print res_sorted,testLabels[testFiles.index(testFile)]
    classified=re.sub('P_','',res_sorted[0][0])
    classified=re.sub('_Vec','',classified)
    if testLabels[testFiles.index(testFile)]!=classified:
        errors += 1
print 'Error ratio is:%f' % (errors/float(len(testFiles)))
