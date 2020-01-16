import numpy as np
from sklearn.utils import shuffle

class Classification():
    def __init__(self, X, t, eta = 0.1, NumSegments = 5, NumEpochs = 10, w0 = None, seed = None):
        self.N = len(t)
        self.D = len(X[0])
        self.NumClass = len(t[0])
        self.t = t
        self.X = X
        self.eta = eta
        self.NumSegments = NumSegments
        self.NumEpochs = NumEpochs
        self.NumEachSegments = int(self.N / self.NumSegments)
        self.w = w0
        if self.w is None:
            if seed is not None:
                np.random.seed(seed)
            self.w = np.random.uniform(-5, 5, self.D*self.NumClass)
        self.w = self.w.reshape(self.D, -1)

    def divSegment(self):
        for i in range(self.NumSegments):
            xSeg = self.X[i*self.NumEachSegments:(i+1)*self.NumEachSegments]
            tSeg = self.t[i*self.NumEachSegments:(i+1)*self.NumEachSegments]
            yield (xSeg, tSeg)

    def sigmoid(self,z):
        return 1.0 / (1.0 + np.exp(-z))

    def softmax(self,Z):
        print("Z.shape: ",Z.shape)
        maxz = np.max(Z, axis=1, keepdims=True)
        Z = Z - maxz
        EZ = np.exp(Z)
        d = np.sum(EZ, axis=1, keepdims=True)
        return EZ/d
    
    def iShuffer(self):
        self.X, self.t = shuffle(self.X, self.t)

    def fit(self):
        for i  in range(self.NumEpochs):
            self._fit()
        return self

    def _fit(self):
        self.iShuffer()
        for xSeg,tSeg in self.divSegment():
            self.__fit(xSeg,tSeg)

    def __fit(self, xSeg, tSeg):
        z = xSeg.dot(self.w)
        yPre = self.sigmoid(z)
        deltaW = xSeg.T.dot(yPre - tSeg)
        self.w = self.w - self.eta*deltaW

    def mse(self,t,ypred):
        N = len(t)
        e = t - ypred
        return e.T.dot(e)/N

    def predict(self, Xpred, t = None):
        prediction = self.sigmoid(Xpred.dot(self.w))
        if t is not None:
            print("MSE = ",format(self.mse(t, prediction)))
        return np.heaviside(prediction - 0.5, 0.0)
    
    def predictNclass(self, Xpred, t = None):
        prediction = self.predictLabel(Xpred, reshape = True)
        if t is not None:
            print("MSE = ",format(self.mse(t, prediction)))
            #print("MSE = ",format(self.mse(t, self.sigmoid(Xpred.dot(self.w)))))
            
        lbl = []
        for i in range(len(prediction)):
            tmp = np.zeros(self.NumClass, dtype = float)
            tmp[int(prediction[i])] = 1.0
            lbl.append(tmp)
        return np.array(lbl) 

    def predictLabel(self, Xpred, reshape = None):
        Z = Xpred.dot(self.w)
        Yh = self.softmax(Z)
        print("yh.shape:",Yh.shape)
        label = np.argmax(Yh, axis = 1)
        if reshape is not None:
            return np.array(label, dtype = float).reshape(-1, 1)
        return np.array(label)
