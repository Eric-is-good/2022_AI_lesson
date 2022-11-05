import torch
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
import torchvision as tv
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
from PIL import Image
from data_loader import creat_dataloader


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    epochs, lr, batch_size = 100, 0.001, 256

    traindata = creat_dataloader("./cifar-10-batches-py/train_batch", batch_size)
    testdata = creat_dataloader("./cifar-10-batches-py/test_batch", batch_size)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = LeNet().cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr, weight_decay=0.0001)
    # optimizer = optim.SGD(net.parameters(), lr)

    for epoch in range(epochs):
        print("epoch is ", epoch)
        running_loss = 0.0
        for data, label in traindata:
            optimizer.zero_grad()
            pre = net(data)
            loss = criterion(pre, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print('loss: %.3f' % (running_loss / (50000 / batch_size)))

        torch.save(net.state_dict(), "./checkpoint/" + str(epoch))

        with torch.no_grad():
            correct = 0
            for data, label in testdata:
                pre = net(data)
                _, predicted = t.max(pre.data, 1)
                _, ground_trues = t.max(label, 1)
                correct += (predicted == ground_trues).sum()
            acc = correct / 10000
            print("acc is ", acc.item())
