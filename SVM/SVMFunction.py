import numpy as np
from sklearn.utils import shuffle
import CommonFunction as cf
from cvxopt import matrix, solvers

class PrimalSVM():
    def __init__(self, X, funcs, t):
        self.Xfeatures = cf.CommonFunction(X, funcs).generate_not_column_one()
        self.M = len(self.Xfeatures[0]) + 1
        self.N = len(self.Xfeatures)
        
        tfinal = t[:, :]
        K = len(self.Xfeatures[0])+1
        tfinal = t[:, :]
        for _ in range(K - 1):
            tfinal = np.hstack((tfinal, t))
        self.t = tfinal
        
        N = len(self.Xfeatures)
        one = np.ones(N, dtype = float).reshape(-1, 1)
        self.Xfeatures = np.hstack((self.Xfeatures, one))
 
    def fit(self):
        K = np.identity(self.M, dtype = float)
        K[self.M - 1][self.M - 1] = 0.0
        K = matrix(K)
        p = matrix(np.zeros(self.M, dtype = float).reshape(-1, 1))
        G = matrix(-(self.Xfeatures * self.t))
        h = -np.ones(self.N, dtype = float).reshape(-1, 1)
        h = matrix(h)

        solvers.options['show_progress'] = False
        solultion = solvers.qp(K ,p ,G , h)
        self.w = np.array(solultion['x']).reshape(-1, 1)
        print("Done!!! \n w = \n{}".format(self.w))
        return self
    
    def add_column_one(self, Xpredict):
        N = len(Xpredict)
        Xpredict = np.hstack((Xpredict, np.ones(N, dtype = float).reshape(-1, 1)))
        return Xpredict
    
    def predict(self, Xpredict):
        Xtmp = self.add_column_one(Xpredict)
        ypredict = np.sum((Xtmp*self.w.T),axis = 1).reshape(-1,1)
        for i in range(len(ypredict)):
            num = np.asscalar(ypredict[i])
            if (num > -1.001 and num < -0.999) or (num < 1.001 and num > 0.999):
                yield Xpredict[i]
        
    def supportVectorPoints(self, Xpredict):
        return np.array([point for point in self.predict(Xpredict)])
    
    def calculate(self, X, t):
        Xtmp = self.add_column_one(X)
        rangePoint = (np.sum((Xtmp*self.w.T),axis = 1)/np.linalg.norm(self.w)).reshape(-1,1)
        heightPoint = t*(np.sum((Xtmp*self.w.T),axis = 1)).reshape(-1,1)
        print("range of Points=\n",rangePoint)
        print("height of Points=\n",heightPoint)
    
    def predict_label(self, Xpredict):
        Xtmp = self.add_column_one(Xpredict)
        ypredict = np.sum((Xtmp*self.w.T),axis = 1).reshape(-1,1)
        for i in range(len(ypredict)):
            if (ypredict[i] > 0):
                ypredict[i] = 1
            else:
                ypredict[i] = -1
        return ypredict
##########################################################################################    
        
class dualSVM():
    def __init__(self, X, funcs, t):
        self.Xfeatures = cf.CommonFunction(X, funcs).generate_not_column_one()
        self.M = len(self.Xfeatures[0]) + 1
        self.N = len(self.Xfeatures)
        self.torig = t
        
    def fit(self):
        Kgram = self.Xfeatures.dot(self.Xfeatures.T)
        Y = self.torig.dot(self.torig.T)
        K = matrix(Kgram * Y)
        p = matrix(-np.ones(self.N).reshape(-1, 1))
        G = matrix(-np.identity(self.N))
        h = matrix(np.zeros(self.N).reshape(-1, 1))
        A = matrix(self.torig.reshape(1, -1))
        b = matrix(np.zeros((1, 1)))
        
        solvers.options['show_progress'] = False
        solultion = solvers.qp(K, p, G, h, A, b)
        alpha = np.array(solultion['x']).reshape(-1, 1)
        self.w = self.calculateW(alpha)
        self.b = self.calculate_b(alpha, self.w, self.torig)
        if self.b is not None:
            print("Done !!! \n, w = \n{}".format(self.w))
            print("b = {}\n".format(self.b))
        return self
    def calculateW(self, alpha):
        tmp = alpha*self.torig
        alt = alpha*self.torig
        for _ in range(1,len(self.Xfeatures[0])):
            alt = np.hstack((alt,tmp))
        
        altx = tmp*self.Xfeatures
        return np.sum(altx, axis = 0).reshape(-1,1)
    
    def calculate_b(self,alpha, w, t):
        for i in range(self.N):
            anum = np.asscalar(alpha[i])
            if anum > 0.01:
                return np.asscalar(t[i] - w.T.dot(self.Xfeatures[i].T))
        return None
           
    def predict(self, Xpredict):
        Npredict = len(Xpredict)
        ytmp = np.sum((Xpredict*self.w.T),axis = 1).reshape(-1,1)
        ypredict = ytmp + self.b*(np.ones(Npredict).reshape(-1, 1))
        for i in range(len(ypredict)):
            num = np.asscalar(ypredict[i])
            if (num > -1.001 and num < -0.999) or (num < 1.001 and num > 0.999):
                yield Xpredict[i]
        
    def supportVectorPoints(self, Xpredict):
        return np.array([point for point in self.predict(Xpredict)])

##########################################################################################    
        
class SoftMargin():
    def __init__(self, X, funcs, t, c):
        self.Xfeatures = cf.CommonFunction(X, funcs).generate_not_column_one()
        self.M = len(self.Xfeatures[0]) + 1
        self.N = len(self.Xfeatures)
        self.torig = t
        self.c = c
        self.funcs = funcs
        
        index = len(self.Xfeatures)
        one = np.ones(index, dtype = float).reshape(-1, 1)
        self.Xfeatures = np.hstack((self.Xfeatures, one))
        
    def fit(self):
        K = self.createK()
        p = self.createP()
        G = self.createG()
        h = self.createH()
        
        solvers.options['show_progress'] = False
        solultion = solvers.qp(K ,p ,G , h)
        alpha = np.array(solultion['x']).reshape(-1, 1)

        self.w = self.calculateW(alpha)
        self.b = self.calculateb(alpha)
        self.E = self.calculateE(alpha)
        return self
    
    def createK(self):
        K = np.zeros((self.M + self.N, self.M + self.N))
        for i in range(self.M-1):
            K[i][i] = 1
        return matrix(K)
    
    def createG(self):
        GM = -(self.Xfeatures * self.torig)
        GN = -(np.identity(self.N, dtype = float))
        Gdum = np.zeros((self.N, self.M))
        GM = np.concatenate((GM, Gdum),axis = 0)
        GN = np.concatenate((GN, GN),axis = 0)
        G  = np.concatenate((GM, GN),axis = 1)
        return matrix(G)
    
    def createP(self):
        Pzeros = np.zeros(self.M)
        Pc = self.c*np.ones(self.N)
        P = np.concatenate((Pzeros, Pc),axis = 0)
        return matrix(P)
    
    def createH(self):
        Hones = -np.ones(self.N)
        Hzeros =  np.zeros(self.N)
        P = np.concatenate((Hones, Hzeros),axis = 0)
        return matrix(P)
    
    def calculateW(self, alpha):
        return alpha[:self.M-1].reshape(-1,1)

    def calculateb(self, alpha):
        return alpha[self.M-1]
                
    def calculateE(self, alpha):
        return alpha[self.M:]
        
    def _predict(self, Xpredict):
        ypredict = self.predict(Xpredict)
        
        for i in range(len(ypredict)):
            num = np.asscalar(ypredict[i])
            if (num > -1.001 and num < -0.999) or (num < 1.001 and num > 0.999):
                yield Xpredict[i]

    def supportVectorPoints(self, Xpredict):
        return np.array([point for point in self._predict(Xpredict)])

    def predict(self, Xpredict):
        Xpre = cf.CommonFunction(Xpredict, self.funcs).generate_not_column_one()
        Npredict = len(Xpre)
        ypredict = Xpre.dot(self.w) + self.b*(np.ones(Npredict).reshape(-1, 1))
        return ypredict
    
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
##########################################################################################    

class DualSoftMargin():
    def __init__(self, X, funcs, t, c):
        self.funcs = funcs
        self.Xfeatures = cf.CommonFunction(X, self.funcs).generate_not_column_one()
        self.M = len(self.Xfeatures[0]) + 1
        self.N = len(self.Xfeatures)
        self.torig = t
        self.c = c
        
    def createK(self):
        Kgram = self.Xfeatures.dot(self.Xfeatures.T)
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
        A = matrix(self.torig.reshape(1, -1))
        print(len(A))
        b = matrix(np.zeros((1, 1)))
        
        solvers.options['show_progress'] = False
        solultion = solvers.qp(K, p, G, h, A, b)
        alpha = np.array(solultion['x']).reshape(-1, 1)
        return alpha
    
    def fit(self):
        alpha = self._fit()
        self.S, self.m = self.splitMS(alpha)
        self.w = self.calculate_w(alpha) 
        self.b = self.calculate_b(alpha)
        return self
        
    def splitMS(self, alpha):
        S = np.where(alpha > 1e-5)[0]
        S2 = np.where(alpha < .99*self.c)[0]
        m = [val for val in S if val in S2] # intersection of two lists              
        return (S,m)

    def calculate_w(self, alpha):
        ts = self.torig[self.S]
        Xs = self.Xfeatures.T[:, self.S]
        Xs = Xs.T
        alphaS = alpha[self.S]
        
        tmp = alphaS*ts
        altx = tmp*Xs
        return np.sum(altx, axis = 0).reshape(-1,1)    
    
    def calculate_b(self,alpha):
        XM = self.Xfeatures.T[:, self.m]
        yM = self.torig[self.m,:].reshape(-1, 1)
        return np.mean(yM - XM.T.dot(self.w))
    
    def predict(self, Xpredict):
        Xpre = cf.CommonFunction(Xpredict, self.funcs).generate_not_column_one()
        Npredict = len(Xpre)
        ypredict = Xpre.dot(self.w) + self.b*(np.ones(Npredict).reshape(-1, 1))
        return ypredict
    
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
    
    def _predict(self, Xpredict):
        ypredict = self.predict(Xpredict)
        
        for i in range(len(ypredict)):
            num = np.asscalar(ypredict[i])
            if (num > -1.0001 and num < -0.9999) or (num < 1.0001 and num > 0.9999):
                yield Xpredict[i]
                
    def supportVectorPoints(self, Xpredict):
        return np.array([point for point in self._predict(Xpredict)])
    