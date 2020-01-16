import numpy as np

class LDAFunction():
    def __init__(self, M):
        self.M = M
        
    def fit_transform(self, Xfeatures):
        Nsample = 210
        subSamples = int(Nsample/3)
        
        X0 = Xfeatures[:subSamples, :]
        N0 = len(X0)
        X1 = Xfeatures[subSamples:subSamples*2, :]
        N1 = len(X1)
        X2 = Xfeatures[subSamples*2:, :]
        N2 = len(X2)
        print("X0:",X0.shape)
        print("X1:",X1.shape)
        print("X2:",X2.shape)
        
        m0 = np.mean(X0, axis = 0, keepdims = True)
        m1 = np.mean(X1, axis = 0, keepdims = True)
        m2 = np.mean(X2, axis = 0, keepdims = True)
        m = np.mean(Xfeatures, axis = 0, keepdims = True)
        
        SB0 = (m0 - m).T.dot(m0 - m)
        SB1 = (m1 - m).T.dot(m1 - m)
        SB2 = (m2 - m).T.dot(m2 - m)
        SB = N0*SB0 + N1*SB1 + N2*SB2
        
        SW0 = (X0 - np.tile(m0, (N0, 1))).T.dot(X0 - np.tile(m0, (N0, 1)))
        SW1 = (X1 - np.tile(m1, (N1, 1))).T.dot(X1 - np.tile(m1, (N1, 1)))
        SW2 = (X2 - np.tile(m2, (N2, 1))).T.dot(X2 - np.tile(m2, (N2, 1)))
        SW = SW0 + SW1 + SW2
        
        A = np.linalg.inv(SW).dot(SB)
        val,vec = np.linalg.eig(A)
        
        Z = Xfeatures - np.tile(m, (Nsample, 1))
        W = vec[:,:self.M]
        return Z.dot(W)