'''

Test kNN on classifying handwriting digits data

'''
from numpy import *
import operator
from os import listdir

import kNN

def img2vector(filename):
	returnVect = zeros((1, 1024))
	fr = open(filename)
	for i in range(32):
		lineStr = fr.readline()
		for j in range(32):
			returnVect[0, 32*i+j] = int(lineStr[j])
	return returnVect


def loadDataSet():
	hwLabels = []
	trainingFileList = listdir('trainingDigits') #load the training set
	m = len(trainingFileList)
	trainingMat = zeros((m,1024))
	for i in range(m):
		fileNameStr = trainingFileList[i]
		fileStr = fileNameStr.split('.')[0]     #take off .txt
		classNumStr = int(fileStr.split('_')[0])
		hwLabels.append(classNumStr)
		trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)

	testLabels = []
	testFileList = listdir('testDigits')        #iterate through the test set
	mTest = len(testFileList)
	testMat = zeros((mTest,1024))
	for i in range(mTest):
		fileNameStr = testFileList[i]
		fileStr = fileNameStr.split('.')[0]     #take off .txt
		classNumStr = int(fileStr.split('_')[0])
		testLabels.append(classNumStr)
		testMat[i,:] = img2vector('testDigits/%s' % fileNameStr)
	return testMat, testLabels, trainingMat, hwLabels

k = 3
testMat, testLabels, trainingMat, hwLabels = loadDataSet()
predictions = kNN.predict(testMat, trainingMat, hwLabels, k)
accuracy =  kNN.accuracy(testLabels, predictions)

print("the total number of test obs is: %d" % len(testLabels))
print("\nthe total accuracy rate is: %f" % accuracy)