from functools import partial
from collections import OrderedDict

from torch import nn

from ark.nn.easy import ConvBnReLU2d


def vgg11(in_channels, num_classes):
    return VGG(in_channels, num_classes, block_depth=[1, 1, 2, 2, 2])


def vgg13(in_channels, num_classes):
    return VGG(in_channels, num_classes, block_depth=[2, 2, 2, 2, 2])


def vgg16(in_channels, num_classes):
    return VGG(in_channels, num_classes, block_depth=[2, 2, 3, 3, 3])


def vgg19(in_channels, num_classes):
    return VGG(in_channels, num_classes, block_depth=[2, 2, 4, 4, 4])


class VGG(nn.Sequential):
    def __init__(self, in_channels, num_classes, block_depth):

        def make_layer(in_channels, out_channels, num_blocks):
            layers = [ConvBlock(in_channels, out_channels)]
            for _ in range(1, num_blocks):
                layers += [ConvBlock(out_channels, out_channels)]
            layers += [nn.MaxPool2d(kernel_size=2)]
            return nn.Sequential(*layers)

        features = nn.Sequential(OrderedDict([
            ('layer1', make_layer(in_channels, 64, block_depth[0])),
            ('layer2', make_layer(64, 128, block_depth[1])),
            ('layer3', make_layer(128, 256, block_depth[2])),
            ('layer4', make_layer(256, 512, block_depth[3])),
            ('layer5', make_layer(512, 512, block_depth[4])),
        ]))

        classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7 * 7 * 512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

        super().__init__(OrderedDict([
            ('features', features),
            ('classifier', classifier),
        ]))


ConvBlock = partial(ConvBnReLU2d, kernel_size=3, padding=1)
