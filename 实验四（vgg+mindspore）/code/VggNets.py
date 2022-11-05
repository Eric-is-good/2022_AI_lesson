import numpy as np
from mindspore import nn, context
from mindspore import Tensor
import mindspore as ms


class VggNet(nn.Cell):
    def __init__(self, net_name='vgg16', num_classes=5, keep_prob=0.5):
        super(VggNet, self).__init__()

        self.cfgs = {
            # 列表中的数字代表卷积层卷积的层数，M 代表 maxpool
            'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512,
                      'M'],
        }

        self.name = net_name
        self.num_classes = num_classes
        self.keep_prob = keep_prob

        self.classifier = nn.SequentialCell(
            nn.Dense(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(keep_prob=self.keep_prob),
            nn.Dense(4096, 4096),
            nn.ReLU(),
            nn.Dropout(keep_prob=self.keep_prob),
            nn.Dense(4096, self.num_classes)
        )

        self.baseline = self.Baseline(cfg=self.cfgs[self.name])

    def Baseline(self, cfg: list):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, pad_mode='pad', padding=1)
                layers += [conv2d, nn.ReLU()]
                in_channels = v
        return nn.SequentialCell(*layers)

    def construct(self, x):
        emb = self.baseline(x)
        emb = emb.reshape(emb.shape[0], 512 * 7 * 7)
        pre = self.classifier(emb)
        return pre


if __name__ == '__main__':
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    a = np.random.randn(10, 3, 224, 224)
    net = VggNet(net_name='vgg16')
    print(net)
    a = Tensor(a, dtype=ms.float32)
    b = net(a)
    print(b.shape)
