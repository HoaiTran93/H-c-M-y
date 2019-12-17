import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import matrix_power
import sys

class Lasso_Linear():
    def __init__(self, X, t, lamda, eta, eps, flimit, steps, func, dfunc):
        self.X = X
        self.t = t
        self.lamda = lamda
        self.eta = eta
        self.eps = eps
        self.steps = steps
        self.flimit = flimit
        self.func = func
        self.dfunc = dfunc
        
        xDegree = len(self.X[0])
        #np.random.seed(0)
        self.w = np.random.uniform(-10.0, 10.0, xDegree).reshape(-1, 1)
        print("init w:",self.w)
        
        self.wSmallest = self.w
        self.fwSmallest = self.func(self.X, self.t, self.lamda, self.w)
        self.smallestAt = 0
        self.wRecord = [self.w]
        self.fwRecord= [self.func(self.X, self.t, self.lamda, self.w)]
        

    def evaluate(self):
        self.w = self.w - self.eta*(self.dfunc(self.X, self.t, self.lamda, self.w))
        fw = self.func(self.X, self.t, self.lamda, self.w)
        self.wRecord.append(self.w)
        self.fwRecord.append(fw)
        if fw < self.fwSmallest:
            self.wSmallest = self.w
            self.fwSmallest = fw
            return (True, self.w)
        return (False, self.w)
        
    def fit(self):
        old_w = self.w
        self.count_step = 0
        for i in range(self.steps):
            self.count_step += 1
            
            smallest,new_w = self.evaluate()
            if smallest:
                self.smallestAt = i
            
            reportSteps = self.steps/100
            if i%reportSteps == 0:
                self.report()
                
            if np.linalg.norm(old_w - new_w) <= self.eps:
                break
            elif self.flimit >= self.fwRecord[-1]:
                break

            old_w = new_w
        
        return self
          
    def report(self):
        print("-----------")
        print("w = \n{}".format(self.wSmallest))
        print("f(w) = {}".format(self.fwSmallest))
        print("at step: {}".format(self.smallestAt))

    def predict(self, Xpre, Ypre):
        Ypre_out = Xpre.dot(self.w)
        meanSquareError = self.MSE(Ypre, Ypre_out)
        return Ypre_out, meanSquareError
    
    def MSE(self, Ypre,Ypre_out):
        N = float(len(Ypre.ravel()))
        e = np.subtract(Ypre, Ypre_out)
        MSE = np.asscalar((e.T.dot(e)/N).ravel())
        return MSE