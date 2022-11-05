import pickle

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


class CifarDataset(Dataset):
    def __init__(self, addr):
        package = unpickle(addr)
        package[b'data'] = np.transpose(package[b'data'], (0, 3, 1, 2))
        self.data = torch.tensor(package[b'data'], dtype=torch.float32) / 255
        self.label = torch.tensor(package[b'labels'], dtype=torch.float32) * 0.8 + 0.1

        self.data = self.data.cuda()
        self.label = self.label.cuda()

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return self.label.shape[0]


def creat_dataloader(addr, batch_size=64):
    mydataset = CifarDataset(addr)
    loader = DataLoader(mydataset, batch_size=batch_size, shuffle=True)
    return loader


if __name__ == '__main__':
    mydataset = CifarDataset("./cifar-10-batches-py/train_batch")
    train_loader = DataLoader(mydataset, batch_size=10, shuffle=True)
    for a, b in train_loader:
        print(a.shape, b.shape)
        break
