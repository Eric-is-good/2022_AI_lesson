import matplotlib.pyplot as plt
import numpy as np
import tqdm
from mpl_toolkits.mplot3d import Axes3D

from numpyNN.baseclass import BaseNetwork
from numpyNN.nn import Sequence, Linear
from numpyNN.activation import Relu, Sigmoid
from numpyNN.loss import MSE
from numpyNN.optimize import SGD


def data_loader(addr):
    datas = []

    with open(addr) as f:
        for line in f.readlines():
            line = line.split()
            line_num = []
            for i in line:
                line_num.append(float(i))
            datas.append(line_num)
    datas = np.asarray(datas)

    mean = datas[:, :-1].mean(0)
    std = datas[:, :-1].std(0)
    datas[:, :-1] = (datas[:, :-1] - mean) / std
    np.random.shuffle(datas)
    return datas


def one_hot(arr, class_num):
    one_hot = np.zeros([arr.shape[0], class_num]) + 0.2
    for i in range(arr.shape[0]):
        one_hot[i][int(arr[i][0])] = 0.8
    return one_hot


class Net(BaseNetwork):
    def __init__(self):
        super(Net, self).__init__()
        self.layers = Sequence(
            Linear(4, 5),
            Relu(),
            Linear(5, 3),
            Sigmoid()
        )
        self.criterion = MSE()

    def parameters(self):
        return self.layers.parameters()

    def forward(self, *x):
        x = x[0]
        return self.layers.forward(x)

    def backward(self, grad=None):
        grad = self.criterion.backward(grad)
        self.layers.backward(grad)


acclist = []
for i in range(10):
    net = Net()
    criterion = net.criterion
    optimizer = SGD(net.parameters(), lr=0.001, momentum=0.9)

    traindatas = data_loader("../../data/Iris-train.txt")
    testdatas = data_loader("../../data/Iris-test.txt")
    x = traindatas[:, :-1]
    y = one_hot(traindatas[:, -1:], 3)

    px, py = [], []
    for i in tqdm.tqdm(range(20000)):
        optimizer.zero_grad()
        pred = net(x)
        loss = criterion(pred, y)
        net.backward()
        optimizer.step()
        px.append(i)
        py.append(loss / 75)

    pre = net(testdatas[:, :-1])
    index = np.argmax(pre, axis=1)
    acc = np.sum(index == testdatas[:, -1]) / testdatas.shape[0]

    print(acc, min(py))

    if acc < 0.9:
        print(pre)
        plt.cla()
        plt.plot(px, py, 'r-', lw=1)
        plt.text(0, 0, 'Loss=%.4f' % min(py), fontdict={'size': 20, 'color': 'red'})
        plt.pause(1)

        break

    acclist.append(acc)

acclist = np.asarray(acclist)
print(acclist.mean(0), acclist.std())
