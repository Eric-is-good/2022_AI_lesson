import numpy as np


class MSE(object):
    def __init__(self):
        self.label = None
        self.pred = None
        self.grad = None
        self.loss = None

    def __call__(self, pred, label):
        return self.forward(pred, label)

    def forward(self, pred, label):
        self.pred, self.label = pred, label
        self.loss = np.sum(0.5 * np.square(self.pred - self.label))
        return self.loss

    def backward(self, grad=None):
        self.grad = (self.pred - self.label)
        ret_grad = np.sum(self.grad, axis=0)
        return np.expand_dims(ret_grad, axis=0)
