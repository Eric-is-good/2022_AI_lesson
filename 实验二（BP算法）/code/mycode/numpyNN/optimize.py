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
            parameters.v_weight = parameters.v_weight * self.momentum - self.lr * parameters.wgrad
            parameters.v_bias = parameters.v_bias * self.momentum - self.lr * parameters.bgrad

            parameters.weight += parameters.v_weight
            parameters.bias += parameters.v_bias


