import pickle
import mindspore.dataset as ds
import mindspore as ms
import mindspore.dataset.vision.c_transforms as CV
import numpy as np
from mindspore import Tensor



def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


class CifarDataset:
    def __init__(self, addr):
        package = unpickle(addr)
        self.data = Tensor(package[b'data'] / 255, ms.float32)
        self.label = Tensor(package[b'labels'].nonzero()[1], ms.int32)

    def __getitem__(self, index):
        index = int(index)
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.label)


def creat_dataset(addr, batch_size=64):
    import mindspore.dataset as ds
    dataset_generator = CifarDataset(addr)
    ds = ds.GeneratorDataset(dataset_generator, ["data", "label"], shuffle=False)
    ds = ds.map(input_columns=["data"], operations=[CV.HWC2CHW()])
    ds = ds.shuffle(buffer_size=64).batch(batch_size, drop_remainder=True)
    return ds


if __name__ == '__main__':
    dataset = creat_dataset("./cifar-10-batches-py/train_batch")
    for a,b in dataset:
        print(a)
        print(b)
        break


