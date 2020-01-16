import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from cvxopt import matrix, solvers

class KernelFunction():
    def __init__(self, gamma=None,r=None, d=None):
        self.gamma = gamma
        self.r = r
        self.d = d

    def polynomial(self):
        return lambda x1,x2: pow((self.r + self.gamma*np.dot(x1,x2.T)),self.d) #x1(a1,b1), x2(a2,b2) (γ < x; x0 > +r)^d;
    
    def rbf(self): 
        return lambda x1,x2: np.exp(-self.gamma*np.sum((x2 - x1)*(x2 - x1)))
    
##################################################
class KernelSVM():
    def __init__(self, X, t, c, func):
        self.X = X
        self.torig = self.transfomT(t)
        self.func = func
        self.N = len(X)
        self.c = c

    def transfomT(self,t):
        for i in range(len(t)):
            if t[i] == 0:
                t[i] = -1
        return t        
                
    def createKgam(self):
        Kgram = np.zeros((self.N,self.N))
        for i in range(self.N):
            Xtmp1 = self.X[i]
            for j in range(self.N):
                Xtmp2 = self.X[j]
                Kgram[i,j] = self.func(Xtmp1,Xtmp2)
        
        return matrix(Kgram)
    
    def createK(self):
        Kgram = self.createKgam()
        Y = self.torig.dot(self.torig.T)
        return matrix(Kgram * Y)
    
    def createG(self):
        Gup = -(np.identity(self.N, dtype = float))
        Gdown = np.identity(self.N, dtype = float)
        G = np.concatenate((Gup, Gdown),axis = 0)
        return matrix(G)
    
    def createH(self):
        Hzeros = np.zeros(self.N)
        Hc = self.c*np.ones(self.N)
        H = np.concatenate((Hzeros, Hc),axis = 0)
        return matrix(H)
    
    def _fit(self):
        K = self.createK()
        p = matrix(-np.ones(self.N).reshape(-1, 1))
        G = self.createG()
        h = self.createH()
        A = matrix(self.torig.reshape(1, -1).astype('double')) #A must be double type
        b = matrix(np.zeros((1, 1)))
        
        solvers.options['show_progress'] = False
        solultion = solvers.qp(K, p, G, h, A, b)
        self.alpha = np.array(solultion['x']).reshape(-1, 1)
        return self.alpha
    
    def fit(self):
        self._fit()
        self.S, self.m = self.splitMS()
        self.w = self.calculate_w() 
        self.b = self.calculate_b()
        print("w=",self.w)
        print("b=",self.b)
        return self
        
    def splitMS(self):
        S = np.where(self.alpha > 1e-5)[0]
        S2 = np.where(self.alpha < .99*self.c)[0]
        m = [val for val in S if val in S2] # intersection of two lists              
        return (S,m)

    def calculate_w(self):
        ts = self.torig[self.S]
        Xs = self.X.T[:, self.S]
        Xs = Xs.T
        alphaS = self.alpha[self.S]
        
        tmp = alphaS*ts
        altx = tmp*Xs
        return np.sum(altx, axis = 0).reshape(-1,1)    
    """
    def calculate_b(self):
        XM = self.X.T[:, self.m]
        yM = self.torig[self.m,:].reshape(-1, 1)
        return np.mean(yM - XM.T.dot(self.w))
    """
    def calculate_b(self):
        XM = self.X.T[:, self.m]
        XM = XM.T
        Xs = self.X.T[:, self.S]
        Xs = Xs.T
        Kms = np.zeros((len(XM),len(Xs)))
        for i in range(len(XM)):
            XMtmp = XM[i]
            for j in range(len(Xs)):
                Xstmp = Xs[j]
                Kms[i,j] = self.func(XMtmp,Xstmp) 
        
        alphaS = self.alpha[self.S]
        ts = self.torig[self.S]
        As = (alphaS*ts).reshape(-1, 1)
        
        yM = self.torig[self.m,:].reshape(-1, 1)
        return np.mean(yM - Kms.dot(As))
    
    def predict(self, Xb):
        Xs = self.X.T[:, self.S]
        Xs  = Xs.T
        Kbs = np.zeros((len(Xb),len(Xs)))
        for i in range(len(Xb)):
            Xbtmp = Xb[i]
            for j in range(len(Xs)):
                Xstmp = Xs[j]
                Kbs[i,j] = self.func(Xbtmp,Xstmp)      
        alphaS = self.alpha[self.S]
        ts = self.torig[self.S]
        As = (alphaS*ts).reshape(-1, 1)
        yPredict = Kbs.dot(As) + self.b*np.ones(len(Xb)).reshape(-1, 1)
        return yPredict    
    
    def predict_label(self, Xpre):
        ypredict = self.predict(Xpre)
        for i in range(len(ypredict)):
            if (ypredict[i] > 0):
                ypredict[i] = 1
            else:
                ypredict[i] = -1
        return ypredict
    
    def _predict(self, Xpredict):
        ypredict = self.predict(Xpredict)
        for i in range(len(ypredict)):
            num = np.asscalar(ypredict[i])
            if (num > -1.0001 and num < -0.9999) or (num < 1.0001 and num > 0.9999):
                yield Xpredict[i]
                
    def supportVectorPoints(self, Xpredict):
        return np.array([point for point in self._predict(Xpredict)])
    
    def classifySign(self,Xpredict,Ypredict):
        minus =[]
        positive = []
        signYpre = np.sign(Ypredict)
        
        for i in range(len(signYpre)):
            if Ypredict[i] > 0:
                positive.append(Xpredict[i])
            else:
                minus.append(Xpredict[i])
        
        print("X(+)=",len(positive))
        print("X(-)=",len(minus))
              
        return (np.array(positive), np.array(minus))