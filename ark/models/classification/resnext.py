from functools import partial
import torch
from torch import nn

from ark.nn.easy import ConvBn2d, ConvBnReLU2d

from .resnet import ResNet


def resnext50_32x4(in_channels, out_channels):
    return ResNeXt(in_channels, out_channels,
                   block_depth=[3, 4, 6, 3],
                   base_width=4,
                   cardinality=32)


def resnext101_32x4(in_channels, out_channels):
    return ResNeXt(in_channels, out_channels,
                   block_depth=[3, 4, 23, 3],
                   base_width=4,
                   cardinality=32)


def resnext101_32x8(in_channels, out_channels):
    return ResNeXt(in_channels, out_channels,
                   block_depth=[3, 4, 23, 3],
                   base_width=8,
                   cardinality=32)


def resnext151_32x4(in_channels, out_channels):
    return ResNeXt(in_channels, out_channels,
                   block_depth=[3, 8, 36, 3],
                   base_width=4,
                   cardinality=32)


class ResNeXt(ResNet):
    def __init__(self, in_channels, out_channels, block_depth, expansion=4, base_width=64, cardinality=1):
        super(ResNeXt, self).__init__(in_channels, out_channels, block_depth,
                                      block=partial(Bottleneck, expansion=expansion,
                                                    base_width=base_width, cardinality=cardinality),
                                      expansion=expansion)


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expansion=4, base_width=64, cardinality=1):
        super(Bottleneck, self).__init__()

        width = int((out_channels / expansion) * (base_width / 64) * cardinality)

        self.conv1 = ConvBnReLU2d(in_channels, width, 1)
        self.conv2 = ConvBnReLU2d(width, width, 3, padding=1, stride=stride, groups=cardinality)
        self.conv3 = ConvBn2d(width, out_channels, 1)

        self.downsample = (
            ConvBn2d(in_channels, out_channels, 1, stride=stride)
            if in_channels != out_channels or stride != 1
            else nn.Identity()
        )

        self.activation = nn.ReLU(inplace=True)

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        residual = self.downsample(input)
        return self.activation(x + residual)
