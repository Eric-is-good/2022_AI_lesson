class BaseNetwork(object):
    def __init__(self):
        pass

    def forward(self, *x):
        pass

    def parameters(self):
        pass

    def backward(self, grad):
        pass

    def __call__(self, *x):
        return self.forward(*x)
