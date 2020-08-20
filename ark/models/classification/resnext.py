from typing import Tuple
from functools import partial
import torch
from torch import nn

from ark.nn.easy import ConvBn2d, ConvBnReLU2d

from .resnet import ResNet

__all__ = [
    'ResNext',
    'resnext50_32x4', 'resnext101_32x4',
    'resnext101_32x8', 'resnext152_32x4'
]


@register_model(
    imagenet1k={'in_channels': 3, 'num_classes': 1000,
                'state_dict': 'https://files.deeplar.tk/ark-weights/resnext50_32x4-imagenet-3c9f5bcfb5.pt'},
)
def resnext50_32x4(in_channels, num_classes):
    return ResNeXt(in_channels, num_classes,
                   block_depth=[3, 4, 6, 3],
                   base_width=4,
                   cardinality=32)


@register_model(
    imagenet1k={'in_channels': 3, 'num_classes': 1000,
                'state_dict': 'https://files.deeplar.tk/ark-weights/resnext101_32x4-imagenet-4d1f34007f.pt'},
)
def resnext101_32x4(in_channels, num_classes):
    return ResNeXt(in_channels, num_classes,
                   block_depth=[3, 4, 23, 3],
                   base_width=4,
                   cardinality=32)


@register_model(
    imagenet1k={'in_channels': 3, 'num_classes': 1000,
                'state_dict': 'https://files.deeplar.tk/ark-weights/resnext101_32x8-imagenet-464ed2f66d.pt'},
)
def resnext101_32x8(in_channels, num_classes):
    return ResNeXt(in_channels, num_classes,
                   block_depth=[3, 4, 23, 3],
                   base_width=8,
                   cardinality=32)


@register_model(
    imagenet1k={'in_channels': 3, 'num_classes': 1000,
                'state_dict': 'https://files.deeplar.tk/ark-weights/resnext152_32x4-imagenet-26b33c40d2.pt'},
)
def resnext152_32x4(in_channels, num_classes):
    return ResNeXt(in_channels, num_classes,
                   block_depth=[3, 8, 36, 3],
                   base_width=4,
                   cardinality=32)


class ResNeXt(ResNet):
    r"""
    """

    def __init__(self, in_channels: int, num_classes: int,
                 block_depth: nn.Module,
                 init_channels: int = 64,
                 block_channels: Tuple[int, int, int, int] = [256, 512, 1024, 2048],
                 expansion: int = 4,
                 base_width: int = 64,
                 cardinality: int = 1):
        super(ResNeXt, self).__init__(
            in_channels, num_classes,
            block=partial(Bottleneck, expansion=expansion, base_width=base_width, cardinality=cardinality),
            block_depth=block_depth,
            init_channels=64,
            block_channels=[256, 512, 1024, 2048],
        )


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
