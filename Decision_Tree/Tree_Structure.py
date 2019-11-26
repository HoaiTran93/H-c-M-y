import numpy as np

class NodeTree(object):
    def __init__(self, index = None, children = [], entropy = 0, high = 0):
        self.index = index           # index of data in this node
        self.entropy = entropy   # entropy, will fill later
        self.high = high       # distance to root node
        self.split_attribute = None # which attribute is chosen, it non-leaf
        self.children = children # list of its child nodes
        self.element = None       # element of split_attribute in children
        self.label = None       # label of node if it is a leaf

    def set_properties(self, split_attribute, element):
        self.split_attribute = split_attribute
        self.element = element

    def set_label(self, label):
        self.label = label