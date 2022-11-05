from mindspore import nn, context, Model

from data_loader import creat_dataset
from mindspore.train.callback import LossMonitor, ModelCheckpoint, CheckpointConfig

from data_loader import creat_dataset
import time


class LeNet5(nn.Cell):
    def __init__(self, num_class=10, num_channel=3):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        self.fc1 = nn.Dense(16 * 5 * 5, 120)
        self.fc2 = nn.Dense(120, 84)
        self.fc3 = nn.Dense(84, num_class)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

    def construct(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    # train_dataset = creat_dataset("./cifar-10-batches-py/train_batch")
    train_dataset = creat_dataset("mini_batch", 100)
    test_dataset = creat_dataset("./cifar-10-batches-py/test_batch")

    net = LeNet5()

    num_epochs, lr = 1, 5e-3

    loss = nn.loss.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    opt = nn.Adam(net.trainable_params(), lr)
    # loss_cb = LossMonitor(per_print_times=train_dataset.get_dataset_size())
    loss_cb = LossMonitor(per_print_times=1)

    model = Model(net, loss, opt, metrics={'acc', 'loss'})

    model_path = "./checkpoint/"
    config_ck = CheckpointConfig(save_checkpoint_steps=train_dataset.get_dataset_size(), keep_checkpoint_max=10)
    ckpoint_cb = ModelCheckpoint(prefix="checkpoint_net", directory=model_path, config=config_ck)

    print("start")
    time1 = time.time()
    model.train(num_epochs, train_dataset, callbacks=[loss_cb, ckpoint_cb], dataset_sink_mode=True)
    time2 = time.time()
    print(time2 - time1)

    # metrics = model.eval(test_dataset, dataset_sink_mode=False)
    # print('Metrics:', metrics)


