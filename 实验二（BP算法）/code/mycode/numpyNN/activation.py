import numpy as np
from baseclass import BaseNetwork


class Relu(BaseNetwork):
    def __init__(self):
        super(Relu, self).__init__()
        self.input = None
        self.output = None

    def forward(self, *x):
        x = x[0]
        self.input = x
        x[self.input <= 0] *= 0
        self.output = x
        return self.output

    def backward(self, grad):
        grad[self.input > 0] *= 1
        grad[self.input <= 0] *= 0
        return grad


class Sigmoid(BaseNetwork):
    def __init__(self):
        super(Sigmoid, self).__init__()
        self.input = None
        self.output = None

    def forward(self, *x):
        x = x[0]
        self.input = x
        self.output = 1 / (1 + np.exp(-self.input))
        return self.output

    def backward(self, grad):
        grad *= self.output * (1 - self.output)
        return grad
