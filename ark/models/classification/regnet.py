from collections import OrderedDict
import torch
from torch import nn

from ark.nn.easy import ConvBn2d, ConvBnReLU2d


def regnetx_002(in_channels, num_classes):
    return RegNet(in_channels, num_classes, [1, 1, 4, 7], [24, 56, 152, 368], 8)


def regnetx_004(in_channels, num_classes):
    return RegNet(in_channels, num_classes, [1, 2, 7, 12], [32, 64, 160, 384], 16)


def regnetx_006(in_channels, num_classes):
    return RegNet(in_channels, num_classes, [1, 3, 5, 7], [48, 96, 240, 528], 24)


def regnetx_008(in_channels, num_classes):
    return RegNet(in_channels, num_classes, [1, 3, 7, 5], [64, 128, 288, 672], 16)


def regnetx_016(in_channels, num_classes):
    return RegNet(in_channels, num_classes, [2, 4, 10, 2], [72, 168, 408, 912], 24)


def regnetx_032(in_channels, num_classes):
    return RegNet(in_channels, num_classes, [2, 6, 15, 2], [96, 192, 432, 1008], 48)


def regnetx_040(in_channels, num_classes):
    return RegNet(in_channels, num_classes, [2, 5, 14, 2], [80, 240, 560, 1360], 40)


def regnetx_064(in_channels, num_classes):
    return RegNet(in_channels, num_classes, [2, 4, 10, 1], [168, 392, 784, 1624], 56)


def regnetx_080(in_channels, num_classes):
    return RegNet(in_channels, num_classes, [2, 5, 15, 1], [80, 240, 720, 1920], 120)


def regnetx_120(in_channels, num_classes):
    return RegNet(in_channels, num_classes, [2, 5, 11, 1], [224, 448, 896, 2240], 112)


def regnetx_160(in_channels, num_classes):
    return RegNet(in_channels, num_classes, [2, 6, 13, 1], [256, 512, 896, 2048], 128)


def regnetx_320(in_channels, num_classes):
    return RegNet(in_channels, num_classes, [2, 7, 13, 1], [336, 672, 1344, 2520], 168)


class RegNet(nn.Sequential):
    def __init__(self, in_channels, num_classes,
                 block_depth, block_channels, group_width=1):
        def make_layer(in_channels, out_channels, num_blocks, stride=2):
            layers = [Bottleneck(in_channels, out_channels, stride=stride, group_width=group_width)]
            for _ in range(1, num_blocks):
                layers += [Bottleneck(out_channels, out_channels, group_width=group_width)]
            return nn.Sequential(*layers)

        features = nn.Sequential(OrderedDict([
            ('head', ConvBnReLU2d(in_channels, 32, 3, padding=1, stride=2)),
            ('layer1', make_layer(32, block_channels[0], block_depth[0])),
            ('layer2', make_layer(block_channels[0], block_channels[1], block_depth[1])),
            ('layer3', make_layer(block_channels[1], block_channels[2], block_depth[2])),
            ('layer4', make_layer(block_channels[2], block_channels[3], block_depth[3])),
        ]))

        classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(),
            nn.Linear(block_channels[-1], num_classes),
        )

        super(RegNet, self).__init__(OrderedDict([
            ('features', features),
            ('classifier', classifier),
        ]))


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, group_width=1):
        super(Bottleneck, self).__init__()

        self.conv1 = ConvBnReLU2d(in_channels, out_channels, 1)
        self.conv2 = ConvBnReLU2d(out_channels, out_channels, 3, padding=1, stride=stride,
                                  groups=out_channels // min(out_channels, group_width))
        self.conv3 = ConvBn2d(out_channels, out_channels, 1)

        self.downsample = (
            ConvBn2d(in_channels, out_channels, 1, stride=stride)
            if stride != 1 or in_channels != out_channels
            else nn.Identity()
        )

        self.activation = nn.ReLU(inplace=True)

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        residual = self.downsample(input)
        return self.activation(x + residual)
