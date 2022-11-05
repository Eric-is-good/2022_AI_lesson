import numpy as np
import torch
from matplotlib import pyplot as plt

from data_loader import creat_dataloader

from train import LeNet

if __name__ == '__main__':
    net = LeNet().cuda()
    net.load_state_dict(torch.load("./checkpoint/Lenet"))
    testdata = creat_dataloader("./cifar-10-batches-py/test_batch", 32)

    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    for data, label in testdata:
        pre = net(data)
        _, predicted = torch.max(pre.data, 1)
        _, ground_trues = torch.max(label, 1)

        plt.figure()
        for i in range(32):
            plt.subplot(4, 8, i + 1)
            if predicted[i] == ground_trues[i]:
                plt.title("{}".format(classes[predicted[i]]), color="green")
            else:
                plt.title("{}".format(classes[predicted[i]]), color="red")
                print("the pic " + str(i) + " " + classes[predicted[i]] + "-->" + classes[ground_trues[i]])
            img = np.transpose(data[i].cpu(), (1, 2, 0))
            plt.imshow(img)
            plt.axis("off")
        plt.show()
        plt.pause(0.1)

        break
