import numpy as np
import torch

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

    def backward(self, grad):  # grad [batch_size, class_num]
        grad = self.output * (1 - self.output) * grad
        return grad


class SoftMax(BaseNetwork):
    def __init__(self):
        super(SoftMax, self).__init__()
        self.input = None
        self.output = None

    def _softmax(self, x):
        self.input = x
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    def forward(self, input):
        self.output = self._softmax(input)
        return self.output

    def backward(self, grad):
        return grad * self.output * (1 - self.output)


if __name__ == '__main__':
    a = np.asarray([[0.5, 0.5, 0.2, 0.5]])
    s = SoftMax()
    b = s.forward(a)
    c = s.backward(np.ones_like(a))
    print(c)

    ta = torch.tensor(a, requires_grad=True)
    tb = torch.softmax(ta, dim=1)
    tb.backward(torch.ones_like(ta))
    print(ta.grad)
