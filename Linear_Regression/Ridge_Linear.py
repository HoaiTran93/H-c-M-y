import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import matrix_power
import sys

class Ridge_Linear():
    def __init__(self, xFeatures, t, lamda):
        self.xFeatures = xFeatures
        self.t = t
        self.lamda = lamda
        self.N = len(xFeatures)
        
    def fit(self):
        Ilamda = self.lamda * np.identity(len(self.xFeatures.T))
        tmp = np.linalg.inv(np.add(Ilamda,self.xFeatures.T.dot(self.xFeatures)))
        self.W = tmp.dot(self.xFeatures.T).dot(self.t)
        #print("W=\n",self.W)
        return self.W
    
    def MSE(self, Ypre,Ypre_out):
        N = float(len(Ypre.ravel()))
        e = np.subtract(Ypre, Ypre_out)
        MSE = np.asscalar((e.T.dot(e)/N).ravel())
        return MSE
    
    def predict(self, Xpre, Ypre):
        Ypre_out = Xpre.dot(self.W)
        meanSquareError = self.MSE(Ypre, Ypre_out)
        return Ypre_out, meanSquareError
        