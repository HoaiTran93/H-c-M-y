import numpy as np
import matplotlib.pyplot as plt

class PCAFunction():
    def __init__(self, M):
        self.M = M
    
    def _fit(self, X):
        self.N = len(X)
        self.X_mean = np.mean(X, keepdims = True)
        self.Z = X.T - np.tile(self.X_mean, (1, self.N))
        self.S = (self.Z.dot(self.Z.T))/self.N
        self.val, self.vec = np.linalg.eig(self.S)
        return (self.val, self.vec)
    
    def getMeigenvector(self, U):
        UM = U[:,:self.M]
        return UM
    
    def fit_transform(self, X):
        lamda,U = self._fit(X)
        self.UM = self.getMeigenvector(U.T)
        Xu = self.Z.T.dot(self.UM)
        return Xu
    
    def inverse_transform(self, Xu):
        Z = Xu.dot(self.UM.T)
        X = Z.T + np.tile(self.X_mean, (1, self.N))
        return X.T
               
        
    
