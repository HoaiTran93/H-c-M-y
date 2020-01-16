import glob
import os.path as osp
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.models import resnet18
from torch.utils.data import Dataset, DataLoader
from time import time

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0,
        #        dilation=1, groups=1, bias=True, padding_mode='zeros')
        # NCHW
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        
        # Linear(in_features, out_features, bias=True)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        
        # MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        self.max_pool = nn.MaxPool2d(2)
        # ReLU(inplace=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Note: the following two ways for max pooling / relu are equivalent.
        # 1) with torch.nn.functional:
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # 2) with torch.nn:
        x = self.relu(self.max_pool(self.conv2_drop(self.conv2(x))))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
class Train_Model():
    def __init__(self, model, data_train, data_test, lr, momentum, save = False):
        self.model = model
        self.save = save
        self.data_test = data_test
        self.data_train = data_train
        self.optimizer = self.optimizers_model(lr, momentum)
        
    def train(self, epoch, save_interval, log_interval = 100):
        self.model.train()
        iteration = 0
        for ep in range(epoch):
            start_time = time()
            for batch_idx, (data, target) in enumerate(self.data_train):
                # data, target = data.to(device), target.to(device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                self.optimizer.step()
                if iteration % log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        ep, batch_idx * len(data), len(self.data_train.dataset),
                        100. * batch_idx / len(self.data_train), loss.item()))
                if iteration % save_interval == 0 and iteration > 0:
                    self.save_checkpoint('./content/mnist-%i.pth' % iteration)
                iteration += 1
            end_time = time()
            print('Time to complete a epoch: {:.2f}s'.format(end_time - start_time))
            self.test()
            self.save_checkpoint('./content/mnist-%i.pth' % iteration)
            
    def test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.data_test:
                # data, target = data.to(device), target.to(device)
                output = self.model(data)
                test_loss += F.nll_loss(output, target, size_average=False).item()
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(self.data_test.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
              test_loss, correct, len(self.data_test.dataset),
              100. * correct / len(self.data_test.dataset)))        
        
    def save_checkpoint(self, checkpoint_path):
        state = { 'state_dict': self.model.state_dict(), 'optimizer' : self.optimizer.state_dict()}
        torch.save(state, checkpoint_path)
        print('Model saved to %s' % checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        state = torch.load(checkpoint_path)
        self.model.load_state_dict(state['state_dict'])
        self.optimizer.load_state_dict(state['optimizer'])
        print('Model loaded from %s' % checkpoint_path)
        
    def optimizers_model(self, lr, momentum):
        return optim.SGD(self.model.parameters(), lr, momentum)