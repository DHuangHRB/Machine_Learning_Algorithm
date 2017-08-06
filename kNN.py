'''
python version: 3.5
To implement k Nearest Neighbors(kNN)

Input:      X_test: vector to compare to existing dataset (NxM)
            X_train: data set of known vectors (NxM)
            y_train: data set labels (1xM vector)
            k: number of neighbors to use for comparison (should be an odd number)
            
Output:     the most popular class label

'''
from numpy import *
import operator
from os import listdir

def classify0(inX, dataSet, labels, k):
	dataSetSize = dataSet.shape[0]
	diffMat = tile(inX, (dataSetSize, 1)) - dataSet
	sqDiffMat = diffMat ** 2
	sqDistances = sqDiffMat.sum(axis = 1)
	distances = sqDistances ** 0.5
	sortedDistIndicies = distances.argsort()
	classCount={}
	for i in range(k):
		voteIlabel = labels[sortedDistIndicies[i]]
		classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
	sortedClassCount = sorted(classCount.iteritems(), key = operator.itemgetter(1), reverse = True)
	return sortedClassCount[0][0]

def predict(X_test, X_train, y_train, k):
	predictions = []
	sizeTest = X_test.shape[0]
	for i in range(sizeTest):
		distances = ((tile(X_test[i], (X_train.shape[0], 1)) - X_train) ** 2).sum(axis = 1) ** 0.5
		sortedDistIndicies = distances.argsort()
		classCount = {}
		for i in range(k):
			voteIlabel = y_train[sortedDistIndicies[i]]
			classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
		sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)
		predictions.append(sortedClassCount[0][0])
	return predictions

def accuracy(y_test, predictions):
	total = len(y_test)
	correct = 0
	for i in range(total):
		if y_test[i] == predictions[i]:
			correct += 1
	return correct / total
