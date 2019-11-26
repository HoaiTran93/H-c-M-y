import numpy as np
from collections import Counter

class ID3_Functions():
    def __init__(self, X, t):
        self.X = X
        self.t = t
        
    def _count_element(self, data):
        count_total = Counter(data)
        count_total = list(count_total.items())
        count_total = np.asarray(count_total)
        count_element_labels = count_total[:,0]
        count_element_values = np.array(count_total[:,1], dtype='float')
        print("count_element_values:",count_element_values)
        prob_element_values = count_element_values / np.sum(count_element_values)
        print("prob_element_values:",prob_element_values)
        return (count_element_labels, prob_element_values)
    
    def cal_entropy(self, data):
        _,prob = self._count_element(data)
        return -np.sum(prob*np.log(prob))
        
    def entropy(self):
        entropy_result = []
        for i in range(self.X.shape[1]):
            name_column = self.X.columns
            data = self.X[name_column[i]]
            tmp_result = self.cal_entropy(data)
            entropy_result.append(tmp_result)
        return entropy_result