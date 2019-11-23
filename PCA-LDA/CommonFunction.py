import numpy as np

class CommonFunction():
    def __init__(self, x, featureFunc):
        self.x = x
        self.N = len(x)
        self.K = len(x[0])
        self.func = featureFunc
        self.features = len(featureFunc)

    def generate(self):
        xFeatures = self.func[0](self.x)
        for i in range(1,self.features):
            xFeatures = np.column_stack((xFeatures, self.func[i](self.x)))
        return xFeatures



