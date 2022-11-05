import mindspore as ms
from mindspore import nn, context, Model, load_checkpoint, load_param_into_net
import mindspore.dataset.vision.c_transforms as CV
import mindspore.dataset.transforms.c_transforms as C
from mindspore.train.callback import LossMonitor, ModelCheckpoint, CheckpointConfig
from mindspore.common.initializer import initializer, Normal
import matplotlib.pyplot as plt
import matplotlib
import os
import numpy as np


def MnistDataset(data_dir, batch_size=256, resize=(28, 28),
                 rescale=1 / (255 * 0.3081), shift=-0.1307 / 0.3081, buffer_size=64):
    ds = ms.dataset.MnistDataset(data_dir)
    ds = ds.map(input_columns=["image"], operations=[CV.Resize(resize), CV.Rescale(rescale, shift), CV.HWC2CHW()])
    ds = ds.map(input_columns=["label"], operations=C.TypeCast(ms.int32))
    ds = ds.shuffle(buffer_size=buffer_size).batch(batch_size, drop_remainder=True)
    return ds


data_train = MnistDataset("./MNIST_Data/train")

for a in data_train:
    print(a)
    break