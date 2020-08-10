from typing import Tuple
from collections import OrderedDict
import torch
from torch import nn
from torch.nn import functional as F
from math import ceil

from ark.nn.easy import ConvBnReLU2d, ConvBn2d
from ark.nn.utils import round_channels

__all__ = ['GhostNet', 'ghostnet_1_0']


def ghostnet_1_0(in_channels, num_classes):
    r"""GhostNet with width 1.0

    See :class:`~ark.models.classification.ghostnet.GhostNet` for details.
    """
    return GhostNet(in_channels, num_classes)


class GhostNet(nn.Sequential):
    r"""Ghostnet implementation from 
    `"GhostNet: More Features from Cheap Operations": 
    <https://arxiv.org/abs/1911.11907>`_ paper.

    Args:
        in_channels (int): the input channels
        num_classes (int): the number of the output classification classes
        width_multiplier (float): the width multiplier hyperparameter. Default: 1
    """

    def __init__(self, in_channels, num_classes, width_multiplier=1.):

        def c(channels):
            return round_channels(width_multiplier * channels,
                                  divisor=8 if width_multiplier >= 0.1 else 4)

        features = nn.Sequential(OrderedDict([
            ('head', ConvBnReLU2d(3, c(16), 3, padding=1, stride=2)),
            ('layer1', nn.Sequential(
                GhostBottleneck(c(16), c(16), c(16)),
                GhostBottleneck(c(16), c(24), c(48), stride=2),
            )),
            ('layer2', nn.Sequential(
                GhostBottleneck(c(24), c(24), c(72)),
                GhostBottleneck(c(24), c(40), c(72), kernel_size=5, stride=2, use_se=True),
            )),
            ('layer3', nn.Sequential(
                GhostBottleneck(c(40), c(40), c(120), kernel_size=5, use_se=True),
                GhostBottleneck(c(40), c(80), c(240), stride=2),
            )),
            ('layer4', nn.Sequential(
                GhostBottleneck(c(80), c(80), c(200)),
                GhostBottleneck(c(80), c(80), c(184)),
                GhostBottleneck(c(80), c(80), c(184)),
                GhostBottleneck(c(80), c(112), c(480), use_se=True),
                GhostBottleneck(c(112), c(112), c(672), use_se=True),
                GhostBottleneck(c(112), c(160), c(672), kernel_size=5, stride=2, use_se=True),
            )),
            ('layer5', nn.Sequential(
                GhostBottleneck(c(160), c(160), c(960), kernel_size=5),
                GhostBottleneck(c(160), c(160), c(960), kernel_size=5, use_se=True),
                GhostBottleneck(c(160), c(160), c(960), kernel_size=5),
                GhostBottleneck(c(160), c(160), c(960), kernel_size=5, use_se=True),
                ConvBnReLU2d(c(160), c(960), 1),
            )),
        ]))

        classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Conv2d(c(960), c(1280), 1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(c(1280), num_classes),
        )

        super(GhostNet, self).__init__(OrderedDict([
            ('features', features),
            ('classifier', classifier),
        ]))


class GhostBottleneck(nn.Module):
    r"""The residual block of the GhostNet.

    Args:
        in_channels (int): the input channels
        out_channels (int): the output channels
        expansion_channels (int): the number of channels in the mid conv
        kernel_size (int): kernel size of the mid conv
        stride (int): the stride of the block
        use_se (bool): include the Squeeze-Expansion block after the 2nd conv
    """

    def __init__(self, in_channels, out_channels, expansion_channels,
                 kernel_size=3,
                 stride=1,
                 use_se=False):
        super(GhostBottleneck, self).__init__()

        self.conv1 = GhostModule(in_channels, expansion_channels)

        self.conv2 = (
            ConvBn2d(expansion_channels, expansion_channels, kernel_size,
                     padding=kernel_size // 2, stride=stride, groups=expansion_channels)
            if stride != 1
            else nn.Identity()
        )

        self.se = SEBlock(expansion_channels, expansion_channels) if use_se else nn.Identity()
        self.conv3 = GhostModule(expansion_channels, out_channels, use_relu=False)

        self.downsample = (
            nn.Sequential(
                ConvBn2d(in_channels, in_channels, kernel_size,
                         padding=kernel_size//2, stride=stride,
                         groups=in_channels),
                ConvBn2d(in_channels, out_channels, 1),
            )
            if in_channels != out_channels or stride != 1
            else None
        )

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.se(x)
        x = self.conv3(x)
        if self.downsample is not None:
            input = self.downsample(input)
        return x + input


class GhostModule(nn.Module):
    r"""GhostModule replaces the 1x1 convs in the bottleneck.

    Args:
        in_channels (int): the input channels
        out_channels (int): the output channels
        kernel_sizes (tuple of int): the kernel sizes for both convs
        reduction (int): hyperparameter that sets the number of channels in
            both mappings.
        use_relu (bool): include a relu in both convs
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_sizes: Tuple[int, int] = (1, 3),
                 stride: int = 1,
                 reduction: int = 2,
                 use_relu: bool = True):
        super(GhostModule, self).__init__()

        init_channels = ceil(out_channels / reduction)
        new_channels = init_channels * (reduction - 1)

        ConvBlock = ConvBnReLU2d if use_relu else ConvBn2d

        self.conv1 = ConvBlock(in_channels, init_channels, kernel_sizes[0], stride=stride,
                               padding=kernel_sizes[0]//2)
        self.conv2 = ConvBlock(init_channels, new_channels, kernel_sizes[1],
                               padding=kernel_sizes[1] // 2,
                               groups=init_channels)

        self.out_channels = out_channels

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x = torch.cat([x1, x2], dim=1)
        return x[:, :self.out_channels, ...]


class SEBlock(nn.Module):
    def __init__(self, in_channels, out_channels, reduction_ratio=4):
        super(SEBlock, self).__init__()

        reduced_channels = round_by(in_channels / reduction_ratio, 4)

        self.conv1 = nn.Conv2d(in_channels, reduced_channels, 1)
        self.activation = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(reduced_channels, out_channels, 1)

    def forward(self, input):
        x = F.adaptive_avg_pool2d(input, 1)
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        return input * F.hardsigmoid(x)
