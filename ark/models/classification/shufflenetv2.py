from typing import Tuple
from collections import OrderedDict

import torch
from torch import nn

from ark.nn.easy import ConvBnReLU2d, ConvBn2d
from ark.nn.functional import channel_shuffle
from ark.utils.hub import register_model

__all__ = [
    'ShuffleNetV2',
    'shufflenetv2_0_5', 'shufflenetv2_1_0',
    'shufflenetv2_1_5', 'shufflenetv2_2_0',
]


def shufflenetv2_0_5(in_channels, num_classes):
    r"""ShuffleNetV2 with width 0.50

    See :class:`~ark.models.classification.shufflenetv2.ShuffleNetV2` for details.
    """
    return ShuffleNetV2(in_channels, num_classes,
                        block_depth=[4, 8, 4],
                        init_channels=24,
                        block_channels=[48, 96, 192, 1024])


def shufflenetv2_1_0(in_channels, num_classes):
    r"""ShuffleNetV2 with width 1.0

    See :class:`~ark.models.classification.shufflenetv2.ShuffleNetV2` for details.
    """
    return ShuffleNetV2(in_channels, num_classes,
                        block_depth=[4, 8, 4],
                        init_channels=24,
                        block_channels=[116, 232, 464, 1024])


def shufflenetv2_1_5(in_channels, num_classes):
    r"""ShuffleNetV2 with width 1.5

    See :class:`~ark.models.classification.shufflenetv2.ShuffleNetV2` for details.
    """
    return ShuffleNetV2(in_channels, num_classes,
                        block_depth=[4, 8, 4],
                        init_channels=24,
                        block_channels=[176, 352, 704, 1024])


def shufflenetv2_2_0(in_channels, num_classes):
    r"""ShuffleNetV2 with width 2.0

    See :class:`~ark.models.classification.shufflenetv2.ShuffleNetV2` for details.
    """
    return ShuffleNetV2(in_channels, num_classes,
                        block_depth=[4, 8, 4],
                        init_channels=24,
                        block_channels=[244, 488, 976, 2048])


class ShuffleNetV2(nn.Sequential):
    r"""ShuffleNetV2 implementation from
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design":
    <https://arxiv.org/abs/1807.11164>`_ paper.

    Args:
        in_channels (int): the input channels
        num_classes (int): the number of the output classification classes
        block_depth (tuple of int): the number of blocks per layer
        init_channels (int): the initial number of channels for the stem
        block_channels (tuple of int): the number of channels per layer
    """

    def __init__(self, in_channels: int, num_classes: int,
                 block_depth: Tuple[int, int, int, int],
                 init_channels: int,
                 block_channels: Tuple[int, int, int, int]):

        def make_layer(in_channels, out_channels, depth, stride=2):
            layers = [InvertedResidualBlock(in_channels, out_channels, stride=stride)]
            for _ in range(1, depth):
                layers += [InvertedResidualBlock(out_channels, out_channels)]
            return nn.Sequential(*layers)

        features = nn.Sequential(OrderedDict([
            ('stem', nn.Sequential(
                ConvBnReLU2d(in_channels, init_channels, 3,
                             padding=1, stride=2),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )),
            ('layer1', make_layer(
                init_channels, block_channels[0], block_depth[0])),
            ('layer2', make_layer(
                block_channels[0], block_channels[1], block_depth[1])),
            ('layer3', make_layer(
                block_channels[1], block_channels[2], block_depth[2])),
            ('tail', ConvBnReLU2d(block_channels[2], block_channels[3], 1)),
        ]))

        classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(block_channels[4], num_classes)
        )

        super(ShuffleNetV2, self).__init__(OrderedDict([
            ('features', features),
            ('classifier', classifier),
        ]))


class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(InvertedResidualBlock, self).__init__()

        branch_channels = out_channels // 2

        self.left = (
            ConvBnReLU2d(in_channels, branch_channels, 3,
                         padding=1, stride=stride,
                         groups=in_channels)
            if stride != 1 else None)

        self.right = nn.Sequential(
            ConvBnReLU2d(
                in_channels if stride > 1 else branch_channels,
                branch_channels, 1),
            ConvBnReLU2d(branch_channels, branch_channels,
                         kernel_size=3, padding=1, stride=stride,
                         groups=branch_channels),
        )

    def forward(self, input):
        if self.left is None:
            left, right = input.chunk(2, dim=1)
            right = self.right(right)
        else:
            left = self.left(input)
            right = self.right(input)

        x = torch.cat([left, right], dim=1)
        return channel_shuffle(x, 2)
