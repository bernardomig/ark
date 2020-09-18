from typing import Tuple
from collections import OrderedDict
import torch
from torch import nn
from torch.nn import functional as F

from ark.nn.easy import ConvBn2d, ConvBnReLU2d
from ark.utils.hub import register_model


@register_model(
    imagenet1k={'in_channels': 3, 'num_classes': 1000,
                'state_dict': 'regnetx_002-imagenet-dc40ef1c36.pt'},
)
def regnetx_002(in_channels, num_classes):
    r"""RegNetX with 200M flops

    See :class:`~ark.models.classification.regnet.RegNet` for details.
    """
    return RegNet(in_channels, num_classes, BottleneckX, [1, 1, 4, 7], [24, 56, 152, 368], 8)


@register_model(
    imagenet1k={'in_channels': 3, 'num_classes': 1000,
                'state_dict': 'regnetx_004-imagenet-f2708d7db0.pt'},
)
def regnetx_004(in_channels, num_classes):
    r"""RegNetX with 400M flops

    See :class:`~ark.models.classification.regnet.RegNet` for details.
    """
    return RegNet(in_channels, num_classes, BottleneckX, [1, 2, 7, 12], [32, 64, 160, 384], 16)


@register_model(
    imagenet1k={'in_channels': 3, 'num_classes': 1000,
                'state_dict': 'regnetx_006-imagenet-e4462f6128.pt'},
)
def regnetx_006(in_channels, num_classes):
    r"""RegNetX with 600M flops

    See :class:`~ark.models.classification.regnet.RegNet` for details.
    """
    return RegNet(in_channels, num_classes, BottleneckX,  [1, 3, 5, 7], [48, 96, 240, 528], 24)


@register_model(
    imagenet1k={'in_channels': 3, 'num_classes': 1000,
                'state_dict': 'regnetx_008-imagenet-e0d97cccbf.pt'},
)
def regnetx_008(in_channels, num_classes):
    r"""RegNetX with 800M flops

    See :class:`~ark.models.classification.regnet.RegNet` for details.
    """
    return RegNet(in_channels, num_classes, BottleneckX,  [1, 3, 7, 5], [64, 128, 288, 672], 16)


@register_model(
    imagenet1k={'in_channels': 3, 'num_classes': 1000,
                'state_dict': 'regnetx_016-imagenet-bc694a4fb3.pt'},
)
def regnetx_016(in_channels, num_classes):
    r"""RegNetX with 1.6G flops

    See :class:`~ark.models.classification.regnet.RegNet` for details.
    """
    return RegNet(in_channels, num_classes, BottleneckX, [2, 4, 10, 2], [72, 168, 408, 912], 24)


@register_model(
    imagenet1k={'in_channels': 3, 'num_classes': 1000,
                'state_dict': 'regnetx_032-imagenet-897d6a187c.pt'},
)
def regnetx_032(in_channels, num_classes):
    r"""RegNetX with 3.2G flops

    See :class:`~ark.models.classification.regnet.RegNet` for details.
    """
    return RegNet(in_channels, num_classes, BottleneckX,  [2, 6, 15, 2], [96, 192, 432, 1008], 48)


@register_model(
    imagenet1k={'in_channels': 3, 'num_classes': 1000,
                'state_dict': 'regnetx_040-imagenet-f5bc702a82.pt'},
)
def regnetx_040(in_channels, num_classes):
    r"""RegNetX with 4.0G flops

    See :class:`~ark.models.classification.regnet.RegNet` for details.
    """
    return RegNet(in_channels, num_classes, BottleneckX, [2, 5, 14, 2], [80, 240, 560, 1360], 40)


@register_model(
    imagenet1k={'in_channels': 3, 'num_classes': 1000,
                'state_dict': 'regnetx_064-imagenet-14880ce230.pt'},
)
def regnetx_064(in_channels, num_classes):
    r"""RegNetX with 6.4G flops

    See :class:`~ark.models.classification.regnet.RegNet` for details.
    """
    return RegNet(in_channels, num_classes, BottleneckX,  [2, 4, 10, 1], [168, 392, 784, 1624], 56)


@register_model(
    imagenet1k={'in_channels': 3, 'num_classes': 1000,
                'state_dict': 'regnetx_080-imagenet-c1ea72c75e.pt'},
)
def regnetx_080(in_channels, num_classes):
    r"""RegNetX with 8.0G flops

    See :class:`~ark.models.classification.regnet.RegNet` for details.
    """
    return RegNet(in_channels, num_classes, BottleneckX, [2, 5, 15, 1], [80, 240, 720, 1920], 120)


@register_model(
    imagenet1k={'in_channels': 3, 'num_classes': 1000,
                'state_dict': 'regnetx_120-imagenet-1d9a4fb38e.pt'},
)
def regnetx_120(in_channels, num_classes):
    r"""RegNetX with 12G flops

    See :class:`~ark.models.classification.regnet.RegNet` for details.
    """
    return RegNet(in_channels, num_classes, BottleneckX,  [2, 5, 11, 1], [224, 448, 896, 2240], 112)


@register_model(
    imagenet1k={'in_channels': 3, 'num_classes': 1000,
                'state_dict': 'regnetx_160-imagenet-27aec19f17.pt'},
)
def regnetx_160(in_channels, num_classes):
    r"""RegNetX with 16G flops

    See :class:`~ark.models.classification.regnet.RegNet` for details.
    """
    return RegNet(in_channels, num_classes, BottleneckX,  [2, 6, 13, 1], [256, 512, 896, 2048], 128)


@register_model(
    imagenet1k={'in_channels': 3, 'num_classes': 1000,
                'state_dict': 'regnetx_320-imagenet-498fddcb01.pt'},
)
def regnetx_320(in_channels, num_classes):
    r"""RegNetX with 32G flops

    See :class:`~ark.models.classification.regnet.RegNet` for details.
    """
    return RegNet(in_channels, num_classes, BottleneckX, [2, 7, 13, 1], [336, 672, 1344, 2520], 168)


@register_model(
    imagenet1k={'in_channels': 3, 'num_classes': 1000,
                'state_dict': 'regnety_002-imagenet-1febafaf91.pt'},
)
def regnety_002(in_channels, num_classes):
    r"""RegNetY with 200M flops

    See :class:`~ark.models.classification.regnet.RegNet` for details.
    """
    return RegNet(in_channels, num_classes, BottleneckY, [1, 1, 4, 7], [24, 56, 152, 368], 8)


@register_model(
    imagenet1k={'in_channels': 3, 'num_classes': 1000,
                'state_dict': 'regnety_004-imagenet-d52958c3f5.pt'},
)
def regnety_004(in_channels, num_classes):
    r"""RegNetY with 400M flops

    See :class:`~ark.models.classification.regnet.RegNet` for details.
    """
    return RegNet(in_channels, num_classes, BottleneckY, [1, 3, 6, 6], [48, 104, 208, 440], 8)


@register_model(
    imagenet1k={'in_channels': 3, 'num_classes': 1000,
                'state_dict': 'regnety_006-imagenet-aec4b5587e.pt'},
)
def regnety_006(in_channels, num_classes):
    r"""RegNetY with 600M flops

    See :class:`~ark.models.classification.regnet.RegNet` for details.
    """
    return RegNet(in_channels, num_classes, BottleneckY, [1, 3, 7, 4], [48, 112, 256, 608], 16)


@register_model(
    imagenet1k={'in_channels': 3, 'num_classes': 1000,
                'state_dict': 'regnety_008-imagenet-8e1cc4c9d6.pt'},
)
def regnety_008(in_channels, num_classes):
    r"""RegNetY with 800M flops

    See :class:`~ark.models.classification.regnet.RegNet` for details.
    """
    return RegNet(in_channels, num_classes, BottleneckY,  [1, 3, 8, 2], [64, 128, 320, 768], 16)


@register_model(
    imagenet1k={'in_channels': 3, 'num_classes': 1000,
                'state_dict': 'regnety_016-imagenet-0a5a3b2871.pt'},
)
def regnety_016(in_channels, num_classes):
    r"""RegNetY with 1.6G flops

    See :class:`~ark.models.classification.regnet.RegNet` for details.
    """
    return RegNet(in_channels, num_classes, BottleneckY, [2, 6, 17, 2], [48, 120, 336, 888], 24)


@register_model(
    imagenet1k={'in_channels': 3, 'num_classes': 1000,
                'state_dict': 'regnety_032-imagenet-86e0086c3c.pt'},
)
def regnety_032(in_channels, num_classes):
    r"""RegNetY with 3.2G flops

    See :class:`~ark.models.classification.regnet.RegNet` for details.
    """
    return RegNet(in_channels, num_classes, BottleneckY,  [2, 5, 13, 1], [72, 216, 576, 1512], 24)


@register_model(
    imagenet1k={'in_channels': 3, 'num_classes': 1000,
                'state_dict': 'regnety_040-imagenet-c9675e1d87.pt'},
)
def regnety_040(in_channels, num_classes):
    r"""RegNetY with 4.0G flops

    See :class:`~ark.models.classification.regnet.RegNet` for details.
    """
    return RegNet(in_channels, num_classes, BottleneckY, [2, 6, 12, 2], [128, 192, 512, 1088], 64)


@register_model(
    imagenet1k={'in_channels': 3, 'num_classes': 1000,
                'state_dict': 'regnety_064-imagenet-547954424d.pt'},
)
def regnety_064(in_channels, num_classes):
    r"""RegNetY with 6.4G flops

    See :class:`~ark.models.classification.regnet.RegNet` for details.
    """
    return RegNet(in_channels, num_classes, BottleneckY,  [2, 7, 14, 2], [144, 288, 576, 1296], 72)


@register_model(
    imagenet1k={'in_channels': 3, 'num_classes': 1000,
                'state_dict': 'regnety_080-imagenet-ed7c3ea940.pt'},
)
def regnety_080(in_channels, num_classes):
    r"""RegNetY with 8.0G flops

    See :class:`~ark.models.classification.regnet.RegNet` for details.
    """
    return RegNet(in_channels, num_classes, BottleneckY, [2, 4, 10, 1], [168, 448, 896, 2016], 56)


@register_model(
    imagenet1k={'in_channels': 3, 'num_classes': 1000,
                'state_dict': 'regnety_120-imagenet-5bbf19ecaa.pt'},
)
def regnety_120(in_channels, num_classes):
    r"""RegNetY with 12G flops

    See :class:`~ark.models.classification.regnet.RegNet` for details.
    """
    return RegNet(in_channels, num_classes, BottleneckY,  [2, 5, 11, 1], [224, 448, 896, 2240], 112)


@register_model(
    imagenet1k={'in_channels': 3, 'num_classes': 1000,
                'state_dict': 'regnety_160-imagenet-fab6e94e80.pt'},
)
def regnety_160(in_channels, num_classes):
    r"""RegNetY with 16G flops

    See :class:`~ark.models.classification.regnet.RegNet` for details.
    """
    return RegNet(in_channels, num_classes, BottleneckY,  [2, 4, 11, 1], [224, 448, 1232, 3024], 112)


@register_model(
    imagenet1k={'in_channels': 3, 'num_classes': 1000,
                'state_dict': 'regnety_320-imagenet-9bce7d869e.pt'},
)
def regnety_320(in_channels, num_classes):
    r"""RegNetY with 32G flops

    See :class:`~ark.models.classification.regnet.RegNet` for details.
    """
    return RegNet(in_channels, num_classes, BottleneckY, [2, 5, 12, 1], [232, 696, 1392, 3712], 232)


class RegNet(nn.Sequential):
    r"""RegNet implementation from 
    `"Designing Network Design Spaces": <https://arxiv.org/abs/2003.13678>` paper.

    Args:
        in_channels (int): the input channels
        num_classes (int): the number of the output classification classes
        block (:obj:~nn.Module): the residual block class
        block_depth (tuple of int): number of blocks per layer
        block_channels (tuple of int): number of channels per layer
        group_width (int): regnet block hyperparameter
    """

    def __init__(self, in_channels: int, num_classes: int,
                 block: nn.Module,
                 block_depth: Tuple[int, int, int, int],
                 block_channels: Tuple[int, int, int, int],
                 group_width: int):
        def make_layer(in_channels, out_channels, num_blocks, stride=2):
            layers = [block(in_channels, out_channels, stride=stride, group_width=group_width)]
            for _ in range(1, num_blocks):
                layers += [block(out_channels, out_channels, group_width=group_width)]
            return nn.Sequential(*layers)

        features = nn.Sequential(OrderedDict([
            ('stem', ConvBnReLU2d(in_channels, 32, 3, padding=1, stride=2)),
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


class BottleneckX(nn.Module):
    r"""Residual bottleneck for regnetx models"""

    def __init__(self, in_channels, out_channels, stride=1, group_width=1):
        super(BottleneckX, self).__init__()

        self.downsample = (
            ConvBn2d(in_channels, out_channels, 1, stride=stride)
            if stride != 1 or in_channels != out_channels
            else nn.Identity()
        )

        self.conv1 = ConvBnReLU2d(in_channels, out_channels, 1)
        self.conv2 = ConvBnReLU2d(out_channels, out_channels, 3, padding=1, stride=stride,
                                  groups=max(out_channels // group_width, 1))

        self.conv3 = ConvBn2d(out_channels, out_channels, 1)

        self.activation = nn.ReLU(inplace=True)

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        residual = self.downsample(input)
        return self.activation(x + residual)


class BottleneckY(nn.Module):
    r"""Residual bottleneck for regnety models"""

    def __init__(self, in_channels, out_channels, stride=1, group_width=1, se_reduction=4):
        super(BottleneckY, self).__init__()

        self.conv1 = ConvBnReLU2d(in_channels, out_channels, 1)
        self.conv2 = ConvBnReLU2d(out_channels, out_channels, 3, padding=1, stride=stride,
                                  groups=out_channels // min(out_channels, group_width))
        self.se = SqueezeExcitation(out_channels, out_channels,
                                    mid_channels=round(in_channels/se_reduction))
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
        x = self.se(x)
        x = self.conv3(x)
        residual = self.downsample(input)
        return self.activation(x + residual)


class SqueezeExcitation(nn.Module):
    r"""Squeeze-excitation block"""

    def __init__(self, in_channels: int, out_channels: int,
                 mid_channels: int):
        super(SqueezeExcitation, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, 1)

    def forward(self, input):
        x = F.adaptive_avg_pool2d(input, 1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return input * torch.sigmoid_(x)
