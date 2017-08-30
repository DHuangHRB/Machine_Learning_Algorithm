'''

Test SVM on classifying handwriting digits data

'''

from numpy import *
import operator
from os import listdir

from svm import svc

def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    fr.close()
    return returnVect
	
def loadImages(dirName):
    hwLabels = []
    fileList = listdir(dirName)
    m = len(fileList)
    dataMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = fileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 9: hwLabels.append(1)
        else: hwLabels.append(-1)
        dataMat[i,:] = img2vector('%s/%s' % (dirName, fileNameStr))
    return dataMat, hwLabels

C = 200
tol = 0.0001
maxIter = 10000
sigma = 10

trainingMat, trainingLabels = loadImages('trainingDigits')
testMat, testLabels = loadImages('testDigits')

clf1 = svc(C, tol, maxIter)
clf1.fit(trainingMat, trainingLabels)
predictions = clf1.predict(trainingMat)
accuracy = clf1.accuracy(trainingLabels, predictions)
print("Training accuracy without kernel: %f" % accuracy)
predictions = clf1.predict(testMat)
accuracy = clf1.accuracy(testLabels, predictions)
print("Test accuracy without kernel: %f" % accuracy)

clf2 = svc(C, tol, maxIter, kernel = 'rbf', sigma = sigma)
clf2.fit(trainingMat, trainingLabels)
predictions = clf2.predict(trainingMat)
accuracy = clf2.accuracy(trainingLabels, predictions)
print("Training accuracy with rbf kernel: %f" % accuracy)
predictions = clf2.predict(testMat)
accuracy = clf2.accuracy(testLabels, predictions)
print("Test accuracy with rbf kernel: %f" % accuracy)

clf3 = svc(C, tol, maxIter, kernel = 'linear')
clf3.fit(trainingMat, trainingLabels)
predictions = clf3.predict(trainingMat)
accuracy = clf3.accuracy(trainingLabels, predictions)
print("Training accuracy with linear kernel: %f" % accuracy)
predictions = clf3.predict(testMat)
accuracy = clf3.accuracy(testLabels, predictions)
print("Test accuracy with linear kernel: %f" % accuracy)
