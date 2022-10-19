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
    return datas


def one_hot(arr, class_num):
    one_hot = np.zeros([arr.shape[0], class_num])
    for i in range(arr.shape[0]):
        one_hot[i][int(arr[i][0])] = 1
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


net = Net()
criterion = net.criterion
optimizer = SGD(net.parameters(), lr=0.001, momentum=0.9)

traindatas = data_loader("../../data/Iris-train.txt")
testdatas = data_loader("../../data/Iris-test.txt")
x = traindatas[:, :-1]
y = one_hot(traindatas[:, -1:], 3)


# px, py = [], []
# for i in tqdm.tqdm(range(5000)):
#     running_loss = 0.0
#     for row in range(x.shape[0]):
#         optimizer.zero_grad()
#         input = x[row:row + 1]
#         label = y[row:row + 1]
#         pred = net(input)
#         loss = criterion(pred, label)
#         running_loss += loss
#         net.backward()
#         optimizer.step()
#     px.append(i)
#     py.append(running_loss/75)
#
#     if i % 100 == 0:
#         print(running_loss/75)
#


px, py = [], []
for i in tqdm.tqdm(range(10000)):
    optimizer.zero_grad()
    pred = net(x)
    loss = criterion(pred, y)
    net.backward()
    optimizer.step()
    px.append(i)
    py.append(loss/75)
    # if i % 1000 == 0:
    #     print(loss/75)

# optimizer = SGD(net.parameters(), lr=0.00001, momentum=0.9)
# for i in tqdm.tqdm(range(10000, 30000)):
#     optimizer.zero_grad()
#     pred = net(x)
#     loss = criterion(pred, y)
#     net.backward()
#     optimizer.step()
#     px.append(i)
#     py.append(loss/75)
    # if i % 1000 == 0:
    #     print(loss/75)


plt.cla()
plt.plot(px, py, 'r-', lw=1)
plt.text(0, 0, 'Loss=%.4f' % min(py), fontdict={'size': 20, 'color': 'red'})
plt.pause(0.1)


pre = net(testdatas[:, :-1])
index = np.argmax(pre, axis=1)
acc = np.sum(index == testdatas[:, -1]) / testdatas.shape[0]
print(acc)
