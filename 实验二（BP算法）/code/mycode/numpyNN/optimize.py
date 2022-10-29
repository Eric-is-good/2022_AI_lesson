class SGD(object):
    def __init__(self, parameters, lr=0.01, momentum=0.9):
        self.parameters = parameters
        self.lr = lr
        self.momentum = momentum
        self.steps = 0

    def zero_grad(self):
        for parameters in self.parameters:
            parameters.wgrad *= 0
            parameters.bgrad *= 0

    def step(self):
        # self.steps += 1
        # if self.steps % 2000 == 0:
        #     self.lr /= 5
        for parameters in self.parameters:
            v_w = parameters.v_weight * self.momentum - self.lr * parameters.wgrad - parameters.v_weight
            v_b = parameters.v_bias * self.momentum - self.lr * parameters.bgrad - parameters.v_bias

            parameters.v_weight += v_w
            parameters.v_bias += v_b

            parameters.weight += parameters.v_weight
            parameters.bias += parameters.v_bias





