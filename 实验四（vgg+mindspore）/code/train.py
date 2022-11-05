
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
from data_loader import creat_dataset
from VggNets import VggNet

num_epochs, lr = 1, 1e-2

if __name__ == '__main__':
    data_train = creat_dataset("../data/flower_photos/train")
    data_test = creat_dataset("../data/flower_photos/test")

    net = VggNet()

    loss = nn.loss.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    opt = nn.Adam(net.trainable_params(), lr)
    # loss_cb = LossMonitor(per_print_times=data_train.get_dataset_size())
    loss_cb = LossMonitor(per_print_times=1)

    model = Model(net, loss, opt, metrics={'acc', 'loss'})

    model_path = "./checkpoint/"
    config_ck = CheckpointConfig(save_checkpoint_steps=data_train.get_dataset_size(), keep_checkpoint_max=10)
    ckpoint_cb = ModelCheckpoint(prefix="checkpoint_net", directory=model_path, config=config_ck)

    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    model.train(num_epochs, data_train, callbacks=[loss_cb, ckpoint_cb], dataset_sink_mode=False)

    metrics = model.eval(data_test, dataset_sink_mode=False)
    print('Metrics:', metrics)