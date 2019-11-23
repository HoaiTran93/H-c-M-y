import numpy as np
import matplotlib.pyplot as plt

class PCAFunction():
    def __init__(self, X, M):
        self.X = X
        self.M = M
        self.N = len(X)
    
    def fit(self):
        self.X_mean = np.mean(self.X, keepdims = True)
        self.Z = self.X.T - np.tile(self.X_mean, (1, self.N))
        self.S = (self.Z.dot(self.Z.T))/self.N
        self.val, self.vec = np.linalg.eig(self.S)
        return (self.val, self.vec)
    
    def getMeigenvector(self, U):
        UM = U[:,:self.M]
        return UM
    
    def reduceDimension(self):
        lamda,U = self.fit()
        self.UM = self.getMeigenvector(U.T)
        Xu = self.Z.T.dot(self.UM)
        return Xu
    
    def recoverDimension(self, Xu):
        Z = Xu.dot(self.UM.T)
        X = Z.T + np.tile(self.X_mean, (1, self.N))
        return X.T
               
        
    
