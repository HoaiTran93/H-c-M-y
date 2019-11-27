import numpy as np
from Tree_Structure import NodeTree

class ID3(object):
    def __init__(self, max_high = 10, min_branch = 2, min_gain = 1e-4):
        self.max_high = max_high
        self.min_branch = min_branch
        self.min_gain = min_gain
        self.root = None
        self.Nsample = 0
        
        
    def fit(self, data, result):
        self.Nsample = data.count()[0]
        self.data = data
        self.attributes = list(data)
        self.result = result
        self.labels = result.unique()
        
        index = range(self.Nsample)
        self.root = NodeTree(index = index, entropy = self.cal_entropy(index), high = 0)
        node_queue = [self.root]
        while node_queue:
            node = node_queue.pop()
            if node.high < self.max_high or node.entropy < self.min_gain:
                node.children = self.split_node(node)
                if not node.children: #leaf node
                    self._set_label(node)
                node_queue += node.children
            else:
                self._set_label(node)
        
        self.print_tree(self.root)
        #return self.root
        
    def cal_entropy(self, index):
        if index == 0:
            return 0
        index = [i+1 for i in index] #pandas index starts from 1
        freq = np.array(self.result[index].value_counts())
        return self.entropy(freq)
    
    def entropy(self, freq):
        freq_0 = freq[np.array(freq).nonzero()[0]] #remove freq = 0  
        prob_0 = freq_0/float(freq_0.sum())
        return -np.sum(prob_0*np.log(prob_0))
    
    def split_node(self, node):
        index = node.index
        best_gain = 0
        best_splits = []
        best_order_splits = []
        best_attributes = None
        element = None
        sub_data = self.data.iloc[index, :]
        for i,attr in enumerate(self.attributes):
            values = self.data.iloc[index, i].unique().tolist()
            if len(values) == 1: #entropy = 0
                continue
            splits = []
            order_splits = []
            for val in values:
                sub_index = sub_data.index[sub_data[attr] == val].tolist()
                splits.append([sub_id-1 for sub_id in sub_index])
                order_splits.append(val)
            # don't split if a node has too small number of points
            if min(map(len, splits)) < self.min_branch: 
                continue
            #information gain
            HxS = 0
            for split in splits:
                HxS += len(split)*self.cal_entropy(split)/len(index)
            gain = node.entropy - HxS
            if gain < self.min_gain: #stop if small gain
                continue
            if gain > best_gain:
                best_gain = gain
                best_splits = splits
                best_attributes = attr
                best_order_splits = order_splits
                element = values
        node.set_properties(best_attributes, element)
        child_nodes = [NodeTree(index = split, order_split_attribute = order_split
                       ,entropy = self.cal_entropy(split), high = node.high + 1) 
                                                        for split, order_split in zip(best_splits, best_order_splits)]
        return child_nodes
    
    def _set_label(self, node):
        target_index = [i+1 for i in node.index]
        node.set_label(self.result[target_index].mode()[0])
        
    def predict(self, pre_data):
        Nsamples = pre_data.count()[0]
        labels = [None]*Nsamples
        for n in range(Nsamples):
            x = pre_data.iloc[n, :]
            # start from root and recursively travel if not meet a leaf
            node = self.root
            while node.children:
                node = node.children[node.element.index(x[node.split_attribute])]
            labels[n] = node.label
            
        return labels
    
    def print_tree(self, node,  file=None, _prefix="", _last=True):
        print(_prefix, "`- " if _last else "|- ", '(',node.order_split_attribute, ')',
              node.split_attribute if node.split_attribute != None else node.label, sep="", file=file)
        _prefix += "   " if _last else "|  "
        child_count = len(node.children)
        for i, child in enumerate(node.children):
            _last = i == (child_count - 1)
            self.print_tree(child, file, _prefix, _last)
            
                