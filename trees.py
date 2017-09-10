'''
python version: 3.5
To implement classic decision tree

Parameters:			
        X_train: array-like training data of nominal values
        y_train: classes of training data of nominal values
        X_test: array-like test data of nominal values 
        y_test: output for classes of test data
		
Key Attributes:
        trees: underlying tree object
		lables: lables of training data 
		
Method:
        fit(X_train, y_train): build a decision tree classifier from the given training data
        predict(X_test): predit class of test data
        accuracy(y_test, predictions): return mean accuracy on the given test data and labels
		dumpTree(filename, trees): store a decision tree classifier on local drive
		loadTree(filename): load a decision tree classifier
'''

import numpy as np
from math import log
import operator
from sklearn.externals import joblib
from collections import Counter

class trees:

    def __init__(self):
        self.trees = {}
        self.labels = []

    def calcEntropy(self, dataSet):
        numEntries = len(dataSet)
        uniqueLabels = np.unique(dataSet[:,-1])
        shannonEnt = 0
        for label in uniqueLabels:
            prob = len(dataSet[dataSet[:,-1] == label]) / numEntries
            shannonEnt -= prob * log(prob, 2)
        return shannonEnt

    def splitDataSet(self, dataSet, axis, value):
        axes = np.shape(dataSet)[1]
        msk = dataSet[:, axis] == value
        colIdx = [col for col in range(axes) if col != axis]
        return dataSet[msk][:, colIdx]

    def chooseBestFeature(self, dataSet):
        numFeatures = np.shape(dataSet)[1] - 1  
        baseEntropy = self.calcEntropy(dataSet)
        bestInfoGain = 0; bestFeature = -1
        for i in range(numFeatures):
            uniqueVals = np.unique(dataSet[:, i])
            newEntropy = 0
            for value in uniqueVals:
                subDataSet = self.splitDataSet(dataSet, i, value)
                prob = len(subDataSet) / len(dataSet)
                newEntropy += prob * self.calcEntropy(subDataSet)     
            infoGain = baseEntropy - newEntropy
            if (infoGain > bestInfoGain): 
                bestInfoGain = infoGain
                bestFeature = i
        return bestFeature
    
    def majorityCnt(self, classList):
        c = Counter(classList)
        return c.most_common(1)[0][0]
    
    def createTree(self, dataSet, labels):
        classList = dataSet[:, -1].tolist()
        if classList.count(classList[0]) == len(classList): 
            return classList[0]
        if len(dataSet[0]) == 1:
            return self.majorityCnt(classList)
        bestFeat = self.chooseBestFeature(dataSet)
        bestFeatLabel = labels[bestFeat]
        trees = {bestFeatLabel:{}}
        subLabels = labels[:]
        del(subLabels[bestFeat])
        uniqueVals = np.unique(dataSet[:, bestFeat])
        for value in uniqueVals:
            trees[bestFeatLabel][value] = self.createTree(self.splitDataSet(dataSet, bestFeat, value), subLabels)
        return trees
        
    def fit(self, X_train, y_train):
        self.labels = ['x' + str(i) for i in range(1, len(X_train[0]) + 1)]
        X_train_arr = np.array(X_train)
        y_train_arr = np.array(y_train)[:, np.newaxis]
        train = np.append(X_train_arr, y_train_arr, axis = 1)
        self.trees = self.createTree(train, self.labels)
    
    def searchTree(self, inputTrees, labels, testVec):
        firstKey = list(inputTrees)[0]
        secondTree = inputTrees[firstKey]
        featIndex = labels.index(firstKey)
        key = testVec[featIndex]
        valueOfFeat = secondTree[key]
        if isinstance(valueOfFeat, dict): 
            classLabel = self.searchTree(valueOfFeat, labels, testVec)
        else: classLabel = valueOfFeat
        return classLabel
        
    def predict(self, X_test):
        predictions = np.array([])
        for each in np.array(X_test):
            pred = self.searchTree(self.trees, self.labels, each)
            predictions = np.append(predictions, pred)
        return predictions
        
    def accuracy(self, y_test, predictions):
        total = len(y_test)
        correct = sum(np.array(y_test).flatten() == predictions)
        return correct / total
    
    def dumpTree(self, filename, trees):
        with open(filename, 'wb') as fw:
            joblib.dump(trees, fw)
            
    def loadTree(self, filename):
        with open(filename, 'rb') as fr:
            return joblib.load(fr)