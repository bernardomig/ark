from typing import Tuple
from functools import partial
from torch import nn
from torch.nn import functional as F

from ark.nn.easy import ConvBnReLU2d, ConvBn2d
from ark.utils.hub import register_model

from .resnet import ResNet


@register_model(
    imagenet1k={'in_channels': 3, 'num_classes': 1000,
                'state_dict': 'wrn50_2_0-imagenet-58c3842933.pt'},
)
def wrn50_2_0(in_channels, num_classes):
    r"""Wide ResNet50 with width multiplier 2.0

    See :class:`~ark.models.classification.wrn.WRN` for details.
    """
    return WRN(in_channels, num_classes,
               block_depth=[3, 4, 6, 3],
               width_multiplier=2.)


@register_model(
    imagenet1k={'in_channels': 3, 'num_classes': 1000,
                'state_dict': 'wrn101_2_0-imagenet-e4c5597950.pt'},
)
def wrn101_2_0(in_channels, num_classes):
    r"""Wide ResNet50 with width multiplier 2.0

    See :class:`~ark.models.classification.wrn.WRN` for details.
    """
    return WRN(in_channels, num_classes,
               block_depth=[3, 4, 23, 3],
               width_multiplier=2.)


class WRN(ResNet):
    r"""Wide ResNets implementation from the 
    `"Wide Residual Networks": <https://arxiv.org/abs/1605.07146>` paper.

    Args:
        in_channels (int): the input channels
        num_classes (int): the number of the output classification classes
        block_depth (tuple of int): number of blocks per layer
        expansion (int): expansion hyperparameter for the residual block
        width_multiplier (int): width scaling hyperparameter
    """

    def __init__(self, in_channels: int, num_classes: int,
                 block_depth: Tuple[int, int, int, int],
                 expansion: int = 4,
                 width_multiplier: float = 1.):
        super().__init__(
            in_channels, num_classes,
            block=partial(Bottleneck, expansion=expansion, width_multiplier=width_multiplier),
            block_depth=block_depth,
            init_channels=64,
            block_channels=[256, 512, 1024, 2048])


class Bottleneck(nn.Module):
    r"""WideResNet residual bottleneck.

    Args:
        stride (int): the stride of the block
        expansion (int): determines the expansion ratio for the bottleneck
        width_multiplier (float): hyperparameter that scales the number of 
            channels in the 2nd conv
    """

    def __init__(self, in_channels: int, out_channels: int,
                 stride: float = 1,
                 expansion: int = 4,
                 width_multiplier: float = 1):
        super(Bottleneck, self).__init__()

        width = round((out_channels / expansion) * width_multiplier)

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
