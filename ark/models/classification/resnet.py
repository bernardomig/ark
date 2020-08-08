from typing import Tuple
from collections import OrderedDict
from torch import nn
from torch.nn import functional as F

from ark.nn.easy import ConvBnReLU2d, ConvBn2d

__all__ = [
    'ResNet',
    'resnet18', 'resnet34',
    'resnet50', 'resnet101', 'resnet152',
]


def resnet18(in_channels, num_classes):
    r"""resnet18 model

    See :class:`~ark.models.classification.resnet.ResNet` for details.
    """
    return ResNet(in_channels, num_classes,
                  block=BasicBlock,
                  block_depth=[2, 2, 2, 2],
                  init_channels=64,
                  block_channels=[64, 128, 256, 512])


def resnet34(in_channels, num_classes):
    r"""resnet34 model

    See :class:`~ark.models.classification.resnet.ResNet` for details.
    """
    return ResNet(in_channels, num_classes,
                  block=BasicBlock,
                  block_depth=[3, 4, 6, 3],
                  init_channels=64,
                  block_channels=[64, 128, 256, 512])


def resnet50(in_channels, num_classes):
    r"""resnet50 model

    See :class:`~ark.models.classification.resnet.ResNet` for details.
    """
    return ResNet(in_channels, num_classes,
                  block=Bottleneck,
                  block_depth=[3, 4, 6, 3],
                  init_channels=64,
                  block_channels=[256, 512, 1024, 2048])


def resnet101(in_channels, num_classes):
    r"""resnet101 model

    See :class:`~ark.models.classification.resnet.ResNet` for details.
    """
    return ResNet(in_channels, num_classes,
                  block=Bottleneck,
                  block_depth=[3, 4, 23, 3],
                  init_channels=64,
                  block_channels=[256, 512, 1024, 2048])


def resnet152(in_channels, num_classes):
    r"""resnet152 model

    See :class:`~ark.models.classification.resnet.ResNet` for details.
    """
    return ResNet(in_channels, num_classes,
                  block=Bottleneck,
                  block_depth=[3, 8, 36, 3],
                  init_channels=64,
                  block_channels=[256, 512, 1024, 2048])


class ResNet(nn.Sequential):
    r"""ResNet implementation from 
    `"Deep Residual Learning for Image Recognition": 
    <https://arxiv.org/abs/1512.03385>`_ paper.

    Args:
        in_channels (int): the input channels
        num_classes (int): the number of the output classification classes
        block (:obj:~nn.Module): the residual block class
        block_depth (tuple of int): number of blocks per layer
        init_channels (int): the input channels of the feature encoder
        block_channels (tuple of int): number of channels per layer
    """

    def __init__(self, in_channels: int, num_classes: int,
                 block: nn.Module,
                 block_depth: Tuple[int, int, int, int],
                 init_channels: int,
                 block_channels: Tuple[int, int, int, int]):

        def make_layer(in_channels, out_channels, num_blocks, stride=2):
            layers = [block(in_channels, out_channels, stride=stride)]
            for _ in range(1, num_blocks):
                layers += [block(out_channels, out_channels)]
            return nn.Sequential(*layers)

        features = nn.Sequential(OrderedDict([
            ('stem', nn.Sequential(OrderedDict([
                ('conv', ConvBnReLU2d(in_channels, init_channels, 7, stride=2, padding=3)),
                ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
            ]))),
            ('layer1', make_layer(init_channels, block_channels[0], block_depth[0], stride=1)),
            ('layer2', make_layer(block_channels[0], block_channels[1], block_depth[1])),
            ('layer3', make_layer(block_channels[1], block_channels[2], block_depth[2])),
            ('layer4', make_layer(block_channels[2], block_channels[3], block_depth[3])),
        ]))

        classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(block_channels[3], num_classes),
        )

        super().__init__(OrderedDict([
            ('features', features),
            ('classifier', classifier),
        ]))


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(BasicBlock, self).__init__()

        self.conv1 = ConvBnReLU2d(in_channels, out_channels, 3, padding=1, stride=stride)
        self.conv2 = ConvBn2d(out_channels, out_channels, 3, padding=1)

        self.downsample = (
            ConvBn2d(in_channels, out_channels, 1, stride=stride)
            if in_channels != out_channels or stride != 1
            else nn.Identity()
        )

        self.activation = nn.ReLU(inplace=True)

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        residual = self.downsample(input)
        return self.activation(x + residual)


class Bottleneck(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 stride: int = 1,
                 expansion: int = 4):
        super(Bottleneck, self).__init__()

        width = out_channels // expansion

        self.conv1 = ConvBnReLU2d(in_channels, width, 1)
        self.conv2 = ConvBnReLU2d(width, width, 3, padding=1, stride=stride)
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
