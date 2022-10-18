import numpy as np
from baseclass import BaseNetwork


class Sequence(BaseNetwork):
    def __init__(self, *layer):
        super(Sequence, self).__init__()
        self.layers = []
        self.parameter = []
        for item in layer:
            self.layers.append(item)

        for layer in self.layers:
            if isinstance(layer, Linear):
                self.parameter.append(layer.parameters())

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, *x):
        x = x[0]
        for layer in self.layers:
            x = layer(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def parameters(self):
        return self.parameter


class Variable(object):
    def __init__(self, weight, wgrad, bias, bgrad):
        self.weight = weight
        self.wgrad = wgrad
        self.v_weight = np.zeros(self.weight.shape)
        self.bias = bias
        self.bgrad = bgrad


class Linear(BaseNetwork):
    def __init__(self, inplanes, outplanes):
        super(Linear, self).__init__()
        self.weight = np.random.rand(inplanes, outplanes) * 2 - 1
        self.bias = np.random.rand(outplanes) * 2 - 1
        self.input = None
        self.output = None
        self.wgrad = np.zeros(self.weight.shape)
        self.bgrad = np.zeros(self.bias.shape)
        self.variable = Variable(self.weight, self.wgrad, self.bias, self.bgrad)

    def parameters(self):
        return self.variable

    def forward(self, *x):
        x = x[0]
        self.input = x
        self.output = np.dot(self.input, self.weight) + self.bias
        return self.output

    def backward(self, grad):
        self.bgrad = grad
        self.wgrad += np.dot(self.input.T, grad)
        grad = np.dot(grad, self.weight.T)
        return grad
