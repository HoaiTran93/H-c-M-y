import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from SVMFunction import DualSoftMargin
from SVMFunction import SoftMargin

class CommonFunction():
    def __init__(self, x, featureFunc):
        self.x = x
        self.N = len(x)
        self.K = len(x[0])
        self.func = featureFunc
        self.features = len(featureFunc)

    def generate(self):
        xFeatures = np.ones((self.N,1))
        for i in range(self.features):
            xFeatures = np.column_stack((xFeatures, self.func[i](self.x)))
        return xFeatures

    def generate_not_column_one(self):
        xFeatures = self.func[0](self.x)
        for i in range(1,self.features):
            xFeatures = np.column_stack((xFeatures, self.func[i](self.x)))
        return xFeatures

    
class plotSVM():
    def __init__(self):
        return None
    
    def plotlineSVM(self, svm, opt = None):
        w = svm.w.ravel()
        if opt is None:
            a, b, c = w
            slope = -a/b
        else:
            a, b = w
            c = svm.b
            slope = -a/b       

        #SVM(+) line
        offset = (1-c)/b

        w_xy = np.array([slope, offset]).reshape(-1, 1)
        Xdraw = np.linspace(0, 4, 100).reshape(-1, 1)
        Xdraw_ = np.hstack((Xdraw, np.ones(100).reshape(-1, 1)))
        ydraw = Xdraw_.dot(w_xy)

        plt.plot(Xdraw, ydraw, label = "SVM(+)")

        #SVM(-) line
        offset = (-1-c)/b

        w_xy = np.array([slope, offset]).reshape(-1, 1)
        Xdraw = np.linspace(0, 6, 100).reshape(-1, 1)
        Xdraw_ = np.hstack((Xdraw, np.ones(100).reshape(-1, 1)))
        ydraw = Xdraw_.dot(w_xy)

        plt.plot(Xdraw, ydraw, label = "SVM(-)")
        return self
    
    def plotDualSoftMargin(self, X0, X1, Nsample, funcs, c):
        X = np.concatenate((X0, X1), axis = 0) # all data 
        t = np.concatenate((np.ones((Nsample, 1)), -1*np.ones((Nsample, 1))), axis = 0) # labels 
        X,t = shuffle(X, t)
        
        svm = DualSoftMargin(X, funcs, t, c).fit()
        
        self.plotline(X0, X1, svm, X)
        
        return svm

    def plotSoftMargin(self, X0, X1, Nsample, funcs, c):
        X = np.concatenate((X0, X1), axis = 0) # all data 
        t = np.concatenate((np.ones((Nsample, 1)), -1*np.ones((Nsample, 1))), axis = 0) # labels 
        X,t = shuffle(X, t)
        
        svm = SoftMargin(X, funcs, t, c).fit()
        
        self.plotline(X0, X1, svm, X)
        
        return svm
    
    def plotline(self, X0, X1, svm, X):
        # plot points
        plt.plot(X0[:, 0], X0[:, 1], 'g^', markersize = 8, alpha = .8, label = "class 1")
        plt.plot(X1[:, 0], X1[:, 1], 'ro', markersize = 8, alpha = .8, label = "class -1")
        plt.axis('equal')
        # axis limits
        plt.ylim(0, 4)
        plt.xlim(0, 5)

        plt.xlabel('$x_1$', fontsize = 20)
        plt.ylabel('$x_2$', fontsize = 20)

        w = svm.w.ravel() 
        b = svm.b

        a, c = w # ax + cy + b = 0 => y = -a/c*x - b/c
        slope = -a/c
        offset = -b/c

        w_xy = np.array([slope, offset]).reshape(-1, 1)
        Xdraw = np.linspace(0, 5, 100).reshape(-1, 1)
        Xdraw_ = np.hstack((Xdraw, np.ones(100).reshape(-1, 1)))
        ydraw = Xdraw_.dot(w_xy)

        plt.plot(Xdraw, ydraw, label = "split")

        sv_points = svm.supportVectorPoints(X)

        if len(sv_points) > 0:
            Xmark = sv_points[:, 0].ravel()
            ymark = sv_points[:, 1].ravel()
            plt.scatter(Xmark, ymark, marker = 's', s=80, facecolors='none', edgecolors='k')
        else:
            print("No points are on SVM")
        
        self.plotlineSVM(svm,opt = True)
        plt.legend()
        plt.show()
        return self
        