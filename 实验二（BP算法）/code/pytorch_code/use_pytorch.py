import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.optim import SGD
from tqdm import tqdm


def data_loader(addr):
    datas = []

    with open(addr) as f:
        for line in f.readlines():
            line = line.split()
            line_num = []
            for i in line:
                line_num.append(float(i))
            datas.append(line_num)
    datas = torch.tensor(datas)
    mean = datas[:, :-1].mean(0)
    std = datas[:, :-1].std(0)
    datas[:, :-1] = (datas[:, :-1] - mean) / std
    return datas


def one_hot(tensor, class_num):
    n = tensor.shape[0]
    onehot = torch.zeros(n, class_num).long()
    return onehot.scatter_(dim=1, index=tensor.long(), src=torch.ones(n, class_num).long()).float()


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        out = F.sigmoid(x)
        return out

acclist = []
for i in range(10):
    traindatas = data_loader("../../data/Iris-train.txt")
    testdatas = data_loader("../../data/Iris-test.txt")
    x = traindatas[:, :-1]
    y = one_hot(traindatas[:, -1:], 3)

    net = Net(n_feature=4, n_hidden=5, n_output=3)
    optimizer = SGD(net.parameters(), lr=0.1)
    loss_func = torch.nn.MSELoss()

    px, py = [], []
    for i in tqdm(range(10000)):
        prediction = net(x)
        loss = loss_func(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        px.append(i)
        py.append(loss.item())

    plt.cla()
    plt.plot(px, py, 'r-', lw=1)
    plt.text(0, 0, 'Loss=%.4f' % min(py), fontdict={'size': 20, 'color': 'red'})
    plt.pause(0.1)

    pre = net(testdatas[:, :-1])
    index = torch.argmax(pre, dim=1)
    acc = torch.sum(index == testdatas[:, -1]) / testdatas.shape[0]

    print(acc)
    acclist.append(acc.numpy())

acclist = np.asarray(acclist)
print(acclist.mean(0))
