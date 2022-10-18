class SGD(object):
    def __init__(self, parameters, lr=0.01, momentum=0.9):
        self.parameters = parameters
        self.lr = lr
        self.momentum = momentum

    def zero_grad(self):
        for parameters in self.parameters:
            parameters.wgrad *= 0
            parameters.bgrad *= 0

    def step(self):
        for parameters in self.parameters:
            v = parameters.v_weight * self.momentum - self.lr * parameters.wgrad
            parameters.weight += v
            parameters.bias -= self.lr * parameters.bgrad

