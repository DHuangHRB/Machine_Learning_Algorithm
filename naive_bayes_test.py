'''

Test Naive Bayes on classifying Spam Emails

'''

import random
import numpy as np
from naive_bayes import *

def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

def bagOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2] 
	
def loadDataset():
	docList = []; classList = []
	for i in range(1, 26):
		wordList = textParse(open('email/spam/%d.txt' % i).read())
		docList.append(wordList)
		classList.append(1)
		
		wordList = textParse(open('email/ham/%d.txt' % i).read())
		docList.append(wordList)
		classList.append(0)
	
	trainingSet = list(range(50)); testSet = []
	vocabList = createVocabList(docList)
		
	for i in range(10):
		randIndex = int(random.uniform(0, len(trainingSet)))
		testSet.append(trainingSet[randIndex])
		del(trainingSet[randIndex])
	
	trainMat = []; trainClasses = []
	testMat = []; testClasses = []
	
	for docIndex in trainingSet:
		trainMat.append(bagOfWords2Vec(vocabList, docList[docIndex]))
		trainClasses.append(classList[docIndex])
		
	for docIndex in testSet:
		testMat.append(bagOfWords2Vec(vocabList, docList[docIndex]))
		testClasses.append(classList[docIndex])
		
	return trainMat, trainClasses, testMat, testClasses

X_train, y_train, X_test, y_test = loadDataset()
clf = naive_bayes()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
accuracy = clf.accuracy(y_test, predictions)

print("the total number of test obs is: %d" % len(y_test))
print("\nthe total accuracy rate is: %f" % accuracy)