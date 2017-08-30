'''
python version: 3.5
To implement SVM - Support Vector Machine

Parameters:			
        X_train: training data
        y_train: classes of training data
        X_test: test data 
        y_test: classes of test data
		
        C: constant of slack variables. default = 1.0
        tol: tolerance. default = 0.001
        maxIter: maximum number of iterations. default = 40
        kernel: kernel type. It must be one of 'linear', 'rbf', 'none'. Default is 'none' without using kernel.
        sigma: sigma for rbf kernel only

Key Attributes:
        svIdx: indices of support vectors
        svs: support vectors
        alphas: Lagrange multipliers
        b: intercept/constant in decision function
		
Method:
        fit(X_train, y_train): fit the SVM model according to the given training data
        predict(X_test): perform classification on test data
        accuracy(y_test, predictions): return mean accuracy on the given test data and labels
'''

from numpy import *
import random

class svc:
    
    def __init__(self, C = 1.0, tol = 0.001, maxIter = 40, kernel = 'none', sigma = 10):
        self.X = matrix([])
        self.labelMat = matrix([])
        self.rows = 0
        self.alphas = matrix([])
        self.b = 0
        self.eCache = matrix([])
        self.C = C
        self.tol = tol
        self.maxIter = maxIter
        self.svIdx = array([])
        self.svs = matrix([])
        self.kMat = matrix([])
        self.kernel = kernel
        self.sigma = sigma
        if self.kernel not in ('linear', 'rbf', 'none'):
            print('Kernel is not available. Non-kernel will be applied.')
    
    def kernelTrans(self, X, A):
        m,n = shape(X)
        K = matrix(zeros((m,1)))
        if self.kernel == 'linear':
            K = X * A.T
        elif self.kernel == 'rbf':
            for j in range(m):
                delta = X[j,:] - A
                K[j] = delta * delta.T
            K = exp(K / (-1 * self.sigma**2))
        return K
    
    def selectJrand(self, i, m):
        j = i
        while(j == i):
            j = int(random.uniform(0, m))
        return j

    def clipAlpha(self, aj, L, H):
        if aj > H:
            aj = H
        if aj < L:
            aj = L
        return aj
    
    def calcEk(self, k):
        if self.kernel in ('linear', 'rbf'):
            fxk = float(multiply(self.alphas, self.labelMat).T * self.kMat[:,k]) + self.b
        else:
            fxk = float(multiply(self.alphas, self.labelMat).T * self.X * self.X[k,:].T) + self.b
        Ek = fxk - float(self.labelMat[k])
        return Ek
    
    def selectJ(self, i, Ei):
        maxK = -1; maxDeltaE = 0; Ej = 0
        self.eCache[i] = [1, Ei]
        validEcacheList = nonzero(self.eCache[:,0].A)[0]
        if len(validEcacheList) > 1:
            for k in validEcacheList:
                if k == i:
                    continue
                Ek = self.calcEk(k)
                deltaE = abs(Ei - Ek)
                if deltaE > maxDeltaE:
                    maxK = k; maxDeltaE = deltaE; Ej = Ek
            return maxK, Ej
        else:
            j = self.selectJrand(i, self.rows)
            Ej = self.calcEk(j)
            return j, Ej
    
    def updateEk(self, k):
        Ek = self.calcEk(k)
        self.eCache[k] = [1, Ek]
            
    def innerL(self, i):
        Ei = self.calcEk(i)
        if ((self.labelMat[i] * Ei < -self.tol) and (self.alphas[i] < self.C)) or \
            ((self.labelMat[i] * Ei > self.tol) and (self.alphas[i] > 0)):
            j, Ej = self.selectJ(i, Ei)
            alphaIold = self.alphas[i].copy(); alphaJold = self.alphas[j].copy()
            if (self.labelMat[i] != self.labelMat[j]):
                L = max(0, alphaJold - alphaIold)
                H = min(self.C, self.C + alphaJold - alphaIold)
            else:
                L = max(0, alphaJold + alphaIold - self.C)
                H = min(self.C, alphaJold + alphaIold)
            if L == H: return 0
            
            if self.kernel in ('linear', 'rbf'):
                eta = 2 * self.kMat[i,j] - self.kMat[i,i] - self.kMat[j,j]
            else:
                eta = 2 * self.X[i,:] * self.X[j,:].T - self.X[i,:] * self.X[i,:].T - self.X[j,:] * self.X[j,:].T
                
            if eta >= 0: return 0
            
            self.alphas[j] -= self.labelMat[j] * (Ei - Ej) / eta
            self.alphas[j] = self.clipAlpha(self.alphas[j], L, H)
            self.updateEk(j)
            if abs(self.alphas[j] - alphaJold) < 0.00001: return 0
            
            self.alphas[i] += self.labelMat[i] * self.labelMat[j] * (alphaJold - self.alphas[j])
            self.updateEk(i)
            
            if self.kernel in ('linear', 'rbf'):
                b1 = self.b - Ei - self.labelMat[i] * (self.alphas[i] - alphaIold) * self.kMat[i,i] - \
                    self.labelMat[j] * (self.alphas[j] - alphaJold) * self.kMat[i,j]
                    
                b2 = self.b - Ej - self.labelMat[i] * (self.alphas[i] - alphaIold) * self.kMat[i,j] - \
                    self.labelMat[j] * (self.alphas[j] - alphaJold) * self.kMat[j,j]
            else:
                b1 = self.b - Ei - self.labelMat[i] * (self.alphas[i] - alphaIold) * self.X[i,:] * self.X[i,:].T - \
                    self.labelMat[j] * (self.alphas[j] - alphaJold) * self.X[i,:] * self.X[j,:].T
                
                b2 = self.b - Ej - self.labelMat[i] * (self.alphas[i] - alphaIold) * self.X[i,:] * self.X[j,:].T - \
                    self.labelMat[j] * (self.alphas[j] - alphaJold) * self.X[j,:] * self.X[j,:].T
                    
            if self.alphas[i] > 0 and self.alphas[i] < self.C: self.b = b1
            elif self.alphas[j] > 0 and self.alphas[j] < self.C: self.b = b2
            else: self.b = (b1 + b2) / 2
            return 1
        else:
            return 0
    
    def fit(self, X_train, y_train):
        self.X = matrix(X_train)
        self.labelMat = matrix(y_train).T
        self.rows = shape(X_train)[0]
        self.alphas = matrix(zeros((self.rows,1)))
        self.eCache = matrix(zeros((self.rows,2)))
        self.kMat = matrix(zeros((self.rows,self.rows)))
        if self.kernel in ('linear', 'rbf'):
            for i in range(self.rows):
                self.kMat[:,i] = self.kernelTrans(self.X, self.X[i,:])
        iter = 0
        entireSet = True; alphaPairsChanged = 0
        while(iter < self.maxIter) and (alphaPairsChanged > 0 or entireSet):
            alphaPairsChanged = 0
            if entireSet:
                for i in range(self.rows):
                    alphaPairsChanged += self.innerL(i)
                iter += 1
            else:
                nonBoundIdx = nonzero((self.alphas.A > 0) * (self.alphas.A < self.C))[0]
                for i in nonBoundIdx:
                    alphaPairsChanged += self.innerL(i)
                iter += 1
            if entireSet == True: entireSet = False
            elif alphaPairsChanged == 0: entireSet = True
        
        self.svIdx = nonzero(self.alphas.A > 0)[0]
        self.svs = self.X[self.svIdx]
		
    def predict(self, X_test):
        if self.kernel in ('linear', 'rbf'):
            kernelEval = matrix(zeros((len(self.svIdx), shape(X_test)[0])))
            for i in range(shape(X_test)[0]):
                kernelEval[:,i] = self.kernelTrans(self.svs, matrix(X_test)[i,:])
            predictions = (multiply(self.alphas[self.svIdx], self.labelMat[self.svIdx]).T * kernelEval + self.b).A[0]
        else:
            predictions = (multiply(self.alphas[self.svIdx], self.labelMat[self.svIdx]).T * self.svs * (matrix(X_test).T) + self.b).A[0]
        predictions[predictions > 0] = 1
        predictions[predictions <= 0] = -1
        return predictions
        
    def accuracy(self, y_test, predictions):
        total = len(y_test)
        correct = sum(y_test == predictions)
        return correct / total
		
