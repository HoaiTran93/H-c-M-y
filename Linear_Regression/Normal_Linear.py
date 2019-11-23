import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import matrix_power
import sys

class Normal_Linear():
    def __init__(self, xFeatures, t):
        self.xFeatures = xFeatures
        self.t = t
        self.N = len(xFeatures)
        
    def fit(self):
        self.W = np.linalg.inv(self.xFeatures.T.dot(self.xFeatures)).dot(self.xFeatures.T).dot(self.t)
        print("W=\n",self.W)
        return self.W
    
    def MSE(self, Ypre,Ypre_out):
        N = float(len(Ypre.ravel()))
        e = np.subtract(Ypre, Ypre_out)
        MSE = np.asscalar((e.T.dot(e)/N).ravel())
        return MSE
    
    def predict(self, Xpre, Ypre):
        Ypre_out = Xpre.dot(self.W)
        print("MSE: ", self.MSE(Ypre, Ypre_out))
        return Ypre_out
        
        