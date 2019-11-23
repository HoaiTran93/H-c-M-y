import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import matrix_power
import sys

class Generate():
    def __init__(self, N, sigma, featureFunc):
        self.N = N
        self.sigma = sigma
        self.func = featureFunc
        self.features = len(featureFunc)
            
    def predict_data(self):
        xgen = np.linspace(0, 2*np.pi, self.N).reshape(-1, 1)
        ynoise = np.random.normal(0, self.sigma, self.N).reshape(-1, 1)
        ygen = np.sin(xgen) + ynoise
        return (xgen,ygen)
        
    def train_data(self):
        xgen = np.random.uniform(0,2*np.pi, self.N).reshape(-1, 1)
        ynoise = np.random.normal(0,self.sigma, self.N).reshape(-1, 1)
        ygen = np.sin(xgen) + ynoise
        return (xgen,ygen)
        
        
    def extract_features(self, xgen):
        N = len(xgen)
        xFeatures = np.ones((N,1))
        for i in range(self.features):
            xFeatures = np.column_stack((xFeatures, self.func[i](xgen)))
        return xFeatures