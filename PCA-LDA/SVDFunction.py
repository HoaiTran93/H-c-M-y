import numpy as np
import matplotlib.pyplot as plt

class SVDFunction():
    def __init__(self, X, M):
        self.X = X
        self.M = M
        self.N = len(X)
    """
    def fit(self):
        W = self.X.dot(self.X.T)
        WT = self.X.T.dot(self.X)
        Stmp,self.U = np.linalg.eig(W)
        _,self.V = np.linalg.eig(WT)
        self.VT = self.V.T
        S = np.zeros(self.X.shape)
        for i in range(len(S[0])):
            if i < len(Stmp):
                S[i,i] = np.sqrt(Stmp[i])
            else:
                break
        self.S = S
        print("U",self.U)
        print("S",self.S)
        print("VT",self.VT)
        return self
    """    
    def reduceDimension_using_func_svd(self):
        self.U, Stmp, self.V = np.linalg.svd(self.X)
        self.VT = self.V.T
        S = np.zeros(self.X.shape)
        S[:self.N, :self.N] = np.diag(Stmp)
        self.S = S
        self.getMvalues()
        print("UM",self.UM.shape)
        print("SM",self.SM.shape)
        print("VTM",self.VTM.shape)
        Xu = self.UM.dot(self.SM.dot(self.VTM))
        return Xu
        
    def getMvalues(self):
        self.UM = self.U[:,:self.M]
        self.SM = self.S[:self.M,:self.M]
        self.VTM = self.VT[:self.M,:]
        return self
    
    def reduceDimension(self):
        self.fit()
        self.getMvalues()
        #print("UM",self.UM.shape)
        #print("SM",self.SM.shape)
        #print("VTM",self.VTM.shape)
        Xu = self.UM.dot(self.SM.dot(self.VTM))
        return Xu
    """
    def recoverDimension(self, Xu):
        Z = Xu.dot(self.UM.T)
        X = Z.T + np.tile(self.X_mean, (1, self.N))
        return X.T
    """          
        
    
