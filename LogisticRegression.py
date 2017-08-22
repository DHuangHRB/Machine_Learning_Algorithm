'''
python version: 3.5
To implement Logistic Regression
			
	    X_test: test data 
        X_train: training data
        y_train: classes of training data
        y_test: classes of test data
            
'''
from numpy import *
import random

class LogisticRegression:

	def __init__(self):
		self.weights_ga = array([])
		self.weights_sga = array([])

	def sigmoid(self, z):
		return 1 / (1 + exp(-z))

	def gradAscent(self, X_train, y_train, alpha = 0.01, maxIter = 150):
		dataMat = matrix(X_train)
		labelMat = matrix(y_train).transpose()
		m, n = shape(dataMat)
		self.weights_ga = ones((n,1))
		for i in range(maxIter):
			yhat = self.sigmoid(dataMat * self.weights_ga)
			error = labelMat - yhat
			self.weights_ga = self.weights_ga + alpha * dataMat.T * error

	def fit(self, X_train, y_train, maxIter = 150):
		dataArr = array(X_train)
		labelArr = array(y_train)
		m, n = shape(X_train)
		self.weights_sga = ones(n)
		for j in range(maxIter):
			dataIndex = list(range(m))
			for i in range(m):
				alpha = 4 /(1 + j + i) + 0.0001
				randIndex = int(random.uniform(0, len(dataIndex)))
				yhat = self.sigmoid(sum(dataArr[randIndex] * self.weights_sga))
				error = labelArr[randIndex] - yhat
				self.weights_sga = self.weights_sga + alpha * error * dataArr[randIndex]
				del(dataIndex[randIndex])

	def predict(self, X_test):
		predictions = []
		yhat = self.sigmoid(matrix(X_test) * (matrix(self.weights_sga).T))
		for p in yhat:
			if p > 0.5:
				predictions.append(1)
			else:
				predictions.append(0)
		return predictions

	def accuracy(self, y_test, predictions):
		total = len(y_test)
		correct = 0
		for i in range(total):
			if y_test[i] == predictions[i]:
				correct += 1
		return correct / total
