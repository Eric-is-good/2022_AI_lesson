import numpy as np
import torch


class SoftMax():
    def __init__(self):
        pass

    def _softmax(self, x):
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    def forward(self, input):
        return self._softmax(input)

    def backward(self, input, grad_output):
        out = self.forward(input)
        ret = []
        for i in range(grad_output.shape[0]):
            softmax_grad = np.diag(out[i]) - np.outer(out[i], out[i])
            ret.append(np.dot(softmax_grad, grad_output[i].T))
        ret = np.array(ret)
        return ret


if __name__ == '__main__':
    a = np.asarray([[1.0, 0.5, 1.0, 1.0]])
    b = SoftMax().backward(a, np.ones_like(a))

    # a = a.reshape([1, 4])
    a = torch.tensor(a, requires_grad=True)
    c = torch.softmax(a, dim=1)
    c.backward(torch.ones_like(a))
    print(a.grad)
    print(b)
