import torch
from pyro.nn import PyroModule, PyroParam
from torch.distributions import constraints

class RVM(PyroModule):
    def __init__(self, kernel, X):
        super().__init__()

        self.kernel = kernel
        self.X = X
        self.relevance = PyroParam(torch.ones(X.shape[0]), constraints.positive)

    def forward(self, l, r):
        kxl = self.kernel(l, self.X)
        krx = self.kernel(self.X, r)
        relevance = torch.diag(self.relevance)
        k = kxl @ relevance @ krx

        return k
