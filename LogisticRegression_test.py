'''

Test Logistic Regression on classifying

'''

import numpy as np
from LogisticRegression import *

def loadDataSet():
	frTrain = open('horseColicTraining.txt'); 
	frTest = open('horseColicTest.txt')
	trainingSet = []; trainingLabels = []
	testSet = []; testLabels = []
	for line in frTrain.readlines():
		currLine = line.strip().split('\t')
		lineArr =[]
		for i in range(21):
			lineArr.append(float(currLine[i]))
		trainingSet.append(lineArr)
		trainingLabels.append(float(currLine[21]))

	for line in frTest.readlines():
		currLine = line.strip().split('\t')
		lineArr =[]
		for i in range(21):
			lineArr.append(float(currLine[i]))
		testSet.append(lineArr)
		testLabels.append(float(currLine[21]))
	frTrain.close()
	frTest.close()
	return trainingSet, trainingLabels, testSet, testLabels

X_train, y_train, X_test, y_test = loadDataSet()
multiTests = 10
accSum = 0
for k in range(multiTests):
	clf = LogisticRegression()
	clf.fit(X_train, y_train)
	predictions = clf.predict(X_test)
	accSum += clf.accuracy(y_test, predictions)
print("After %d iterations the average acc rate is: %f" % (multiTests, accSum / multiTests))
