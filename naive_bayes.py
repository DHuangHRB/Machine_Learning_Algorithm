
'''
python version: 3.5
To implement Naive Bayes(NB)
			
	    X_test: list of words vectors of test documents
            X_train: list of words vectors of training documents
            y_train: list of classes of training documents
            y_test: list of classes of test documents
            
'''


import numpy as np

class naive_bayes:
    
    def __init__(self):
        self.p0Vec = np.array([])
        self.p1Vec = np.array([])
        self.p1 = 0
        
    def fit(self, X_train, y_train):
        numTrainDoc = len(X_train)
        numWords = len(X_train[0])
        p0Vec = np.ones(numWords); p1Vec = np.ones(numWords)
        p0Num = 2; p1Num = 2
        for i in range(numTrainDoc):
            if y_train[i] == 1:
                p1Vec += X_train[i]
                p1Num += sum(X_train[i])
            else:
                p0Vec += X_train[i]
                p0Num += sum(X_train[i])
        
        self.p0Vec = np.log(p0Vec / p0Num)
        self.p1Vec = np.log(p1Vec / p1Num)
        self.p1 = sum(y_train) / numTrainDoc

    def predict(self, X_test):
        predictions = []
        for i in range(len(X_test)):
            p1 = sum(X_test[i] * self.p1Vec) + np.log(self.p1)
            p0 = sum(X_test[i] * self.p0Vec) + np.log(1 - self.p1)
            if p1 > p0:
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
