import numpy as np

class GenerateData():
    def __init__(self, pStateBegin, pState, pColorBall):
        self.red = 0
        self.blue = 1
        self.yellow = 2
        self.Color = [self.red, self.blue, self.yellow]
        self.ColorLabel = ['red', 'blue', 'yellow']
        
        self.pStateBegin = pStateBegin
        self.pState = pState
        self.pColorBall = pColorBall
        self.NumState = len(self.pStateBegin[0])


    def generateSateBegin(self):
        self.StateBegin = np.random.choice(np.arange(self.NumState),p=self.pStateBegin[0])
        return self.StateBegin
    
    def generateState(self):
        self.State = np.random.choice(np.arange(self.NumState),p=self.pState[self.StateBegin])
        self.StateBegin = self.State
        return self.State
    
    def generateColorBall(self, State):
        self.colorBall = np.random.choice(self.Color,p=self.pColorBall[State])
        return self.colorBall
    
    def convertColorLabel(self, State, colorBall):
        print("At state: ",State)
        print("Pick ball: ",self.ColorLabel[colorBall])
        return self
    
    def generateRandomData(self,N):
        print("=====Starting...=====")
        self.generateSateBegin()
        self.generateColorBall(self.StateBegin)
        self.convertColorLabel(self.StateBegin, self.colorBall)
        data_arr = np.array([[self.StateBegin], [self.colorBall]])
        print("=====Started========")
        for _ in range(N):
            self.generateState()
            self.generateColorBall(self.State)
            self.convertColorLabel(self.State, self.colorBall)
            tmp = np.array([[self.State], [self.colorBall]])
            data_arr = np.concatenate((data_arr,tmp),axis=1)
        return data_arr
    """
    def softmax(self,X):
        Z = self.seperateLabel(3,X)
        print("Z: ",Z)
        maxz = np.max(Z, axis=1, keepdims=True)
        Z = Z - maxz
        EZ = np.exp(Z)
        d = np.sum(EZ, axis=1, keepdims=True)
        return EZ/d
    """
class GenerateProbability():
    def __init__(self, NumState, NumColorBall):
        self.NumState = NumState
        self.NumColorBall = NumColorBall
        self.Nsample = 100
    def genProbability(self, NState, t):
        ret = []
        for j in range(NState):
            ret.append([])
        for i in range(len(t)):
            ret[int(t[i]) - 1].append(t[i])
        return self.calculateProbability(NState, ret, t)

    def calculateProbability(self, NState, ret, t):
        p = []
        for i in range(NState):
            X = np.array(ret[i])
            tmp = len(X)/len(t)
            p.append(tmp)
        return np.array(p).reshape(1,-1)
    
    def genProbStateBegin(self):
        data = np.random.choice(np.arange(1, self.NumState + 1),self.Nsample)
        pStateBegin = self.genProbability(self.NumState, data)
        return pStateBegin
    
    def genProbState(self):
        data = np.random.choice(np.arange(1, self.NumState + 1),self.Nsample)
        A = self.genProbability(self.NumState, data)
        
        for i in range(1, self.NumState):
            data = np.random.choice(np.arange(1, self.NumState + 1),self.Nsample)
            pState = self.genProbability(self.NumState, data)
            A = np.concatenate((A, pState),axis = 0)
        return A
    
    def genProbBallColor(self):
        data = np.random.choice(np.arange(1, self.NumColorBall + 1),self.Nsample)
        B = self.genProbability(self.NumColorBall, data)
        
        for i in range(1, self.NumState):
            data = np.random.choice(np.arange(1, self.NumColorBall + 1),self.Nsample)
            pState = self.genProbability(self.NumColorBall, data)
            B = np.concatenate((B, pState),axis = 0)
        return B
    
    
    


