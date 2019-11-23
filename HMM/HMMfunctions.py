import numpy as np

class HMM_Function():
    def __init__(self, pStateBegin, pState, pColorball, observations):
        self.pStateBegin = pStateBegin
        self.pState = pState
        self.pColorball = pColorball
        self.observations = observations
        
    def calculate_Problem1(self):
        #initialization
        p = self.pStateBegin*self.pColorball[:,self.observations[0]].reshape(1,-1)
        print(p)
        #Recursion
        for i in range(1,len(self.observations)):
            p = p.dot(self.pState)*self.pColorball[:,self.observations[i]].reshape(1,-1)
            print(p)
        return np.sum(p)
     
    def calculate_Problem2(self):
        #initialization with red
        p1 = self.pStateBegin.T*self.pColorball[:,self.observations[0]].reshape(-1,1)
        p_out = p1
        W = np.array([[0],[0],[0]])
        #Recursion
        p_matrix = p1*self.pState
        for i in range(1,len(self.observations)):
            p = np.max(p_matrix, axis=0).reshape(-1,1)
            p = p*self.pColorball[:,self.observations[i]].reshape(-1,1)
            W_tmp =  np.argmax(p_matrix, axis=0).reshape(-1,1)
            W = np.concatenate((W, W_tmp),axis = 1)
            p_matrix = p*self.pState
            p_out = np.concatenate((p_out, p),axis = 1)
            
        #Path
        print(p_out)
        print(W)
        state = []
        stt = np.argmax(p)
        num = W[stt, len(W[0])-1]
        state.append(num)
        for i in range(len(W[0]) - 1, 0, -1):
            i = i-1
            tmp = W[num, i]
            state.append(tmp)
            num = tmp

        return state
    
    def calA(self):
        #initialization
        p = self.pStateBegin.T*self.pColorball[:,self.observations[0]].reshape(-1,1)
        alpha = p
        print(alpha)
        #Recursion
        for i in range(1,len(self.observations)):
            p = p*self.pState.dot(self.pColorball[:,self.observations[i]].reshape(-1,1))
            alpha = np.concatenate((alpha, p),axis = 1)
        return alpha
    
    def calB(self):
        #initialization
        Blast = np.array([[1],[1],[1]])
        belta = Blast
        #Recursion        
        for i in range(len(self.observations)-1, 0, -1):
            p = self.pState.dot(Blast*self.pColorball[:,self.observations[i]].reshape(-1,1))
            belta = np.concatenate((belta, p),axis = 1)
            Blast = p
        return self.revertMatrix(belta)
    
    def revertMatrix(self, M):
        M_out = M[:,len(M[0])-1].reshape(-1,1)
        for i in range(len(M[0])-1, 0, -1):
            i = i - 1
            M_out = np.concatenate((M_out, M[:,i].reshape(-1,1)),axis = 1)
        return M_out
            
        
class HMMReestimation():
    def __init__(self, A = None, B = None, pi = None, NStates = None, NColorBall = None, Observation = None):
        self.A = A
        self.B = B
        self.pi = pi
        self.Observation = Observation
        self.NStates = NStates
        self.NColorBall = NColorBall
        self.alpha = None
        self.beta = None
        self.theta = None
        self.eta = None
    
    def print_result(self):
        print("A: \n", self.A.round(4))
        print("B: \n", self.B.round(4))
        print("pi: \n", self.pi.round(4))
        
    def fit(self, Nsteps, epsilon):
        observation = self.Observation[1, :]
        for i in range(Nsteps):
            if self._fitEpoch(observation, epsilon):
                print("Reesimattion done at step = {}".format(i))
                self.print_result()
                return self
        print("out of steps")
        return self
    
    def _fitEpoch(self, observation, epsilon):
        self.NObservation = len(observation)
        preA = np.copy(self.A)
        preB = np.copy(self.B)
        prePi = np.copy(self.pi)
        self._InitElement(observation)
        self._fitAlphaBelta(observation)
        self._fitEtaTheta(observation)
        self._UpdateElement(observation)
        Eva_A = self.Evaluation(preA, self.A)
        Eva_B = self.Evaluation(preB, self.B)
        Eva_pi = self.Evaluation(prePi, self.pi)
        return Eva_A < epsilon and Eva_B < epsilon and Eva_pi < epsilon
    
    def _InitElement(self, observation):
        obs = observation[0]
        self.alpha = np.zeros(shape = (self.NStates, self.NObservation))
        self.beta = np.zeros(shape = (self.NStates, self.NObservation))
        self.theta = np.zeros(shape = (self.NStates, self.NObservation))
        self.eta = np.zeros(shape = (self.NStates, self.NStates, self.NObservation - 1))
        self.alpha[:, 0] = self.pi * self.B[:, obs].ravel()
        self.beta[:, -1] = np.ones(self.NStates)        
    
    def _fitAlphaBelta(self, observation):
        self.calAlpha(observation)
        self.calBelta(observation)
        
    def calAlpha(self, observation):
        Obegin = observation[0]
        self.alpha[:, 0] = self.pi * self.B[:, Obegin].ravel()
        for t in range(1, self.NObservation):
            preProb = self.alpha[:, t - 1].reshape(1, -1)
            postProb = preProb.dot(self.A).reshape(-1, 1)
            obsProb = self.B[:, observation[t]].reshape(-1, 1)
            self.alpha[:, t] = (postProb * obsProb).flatten()
            
    def calBelta(self, observation):
        #Obegin = observation[0]
        self.beta[:, -1] = np.ones(self.NStates)
        for t in range(self.NObservation - 2, -1, -1):
            tmp = (self.B[:, observation[t + 1]].ravel() * self.beta[:, t + 1].ravel()).reshape(-1, 1)
            self.beta[:, t] = self.A.dot(tmp).flatten()
    
    def _fitEtaTheta(self, observation):
        for t in range(self.NObservation - 1):
            alpha_t = self.alpha[:, t].reshape(-1, 1)
            belta_t1 = self.beta[:, t+1].reshape(1, -1)
            tmp = alpha_t.dot(belta_t1) * self.A
            obs_t1 = observation[t + 1]
            bj = self.B[:, obs_t1].reshape(1, -1)
            eta_t = tmp * bj
            sumEta = np.sum(eta_t)
            eta_t = eta_t / sumEta
            self.eta[:, :, t] = eta_t
            self.theta[:, t] = np.sum(eta_t, axis=1).ravel()
        self.theta[:, -1] = self.alpha[:, -1] / np.sum(self.alpha[:, -1])
        
    def _UpdateElement(self, observation):
        self.pi = self.theta[:, 0].ravel()
        sum_eta = np.sum(self.eta, axis=2)
        sum_theta = np.sum(self.theta, axis=1)
        sum_theta_t_1 = sum_theta - self.theta[:, -1]
        self.A = (sum_eta.T / sum_theta_t_1).T
        for i in range(self.NColorBall):
            exist = (observation == i).reshape(1, -1)
            theta_obs_i = self.theta * exist
            sum_theta_obs_i = np.sum(theta_obs_i, axis = 1)
            self.B[:, i] = sum_theta_obs_i / sum_theta
            
        
    def Evaluation(self, e1, e2):
        e = e1 - e2
        e = np.power(e, 2)
        return np.sqrt(np.sum(e))
        
            
    


