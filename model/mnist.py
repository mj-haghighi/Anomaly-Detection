import torch.nn as nn
import torch.nn.functional as F

class MNIST(nn.Module):
    def __init__(self):
        super(MNIST, self).__init__()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(784,250)
        self.linear2 = nn.Linear(250,100)
        self.linear3 = nn.Linear(100,10)
    
    def forward(self,X):
        X = self.flatten(X)
        X = self.linear1(X)
        X = self.relu(X)
        X = self.linear2(X)
        X = self.relu(X)
        X = self.linear3(X)
        X = self.relu(X)
        return X