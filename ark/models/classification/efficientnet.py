from collections import OrderedDict
from math import ceil
import torch
from torch import nn
from torch.nn import functional as F

from ark.nn import Swish
from ark.nn.utils import round_channels
from ark.nn.easy import ConvBn2d, ConvBnSwish2d
from ark.utils.hub import register_model

__all__ = [
    'EfficientNet',
    'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2',
    'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5',
    'efficientnet_b6', 'efficientnet_b7',
]


@register_model(
    imagenet1k={'in_channels': 3, 'num_classes': 1000,
                'state_dict': 'efficientnet_b0-imagenet-b28f86c117.pt'},
)
def efficientnet_b0(in_channels, num_classes):
    r"""EfficientNet B0

    See :class:`~ark.models.classification.efficientnet.EfficientNet` for details.
    """
    return EfficientNet(in_channels, num_classes)


@register_model(
    imagenet1k={'in_channels': 3, 'num_classes': 1000,
                'state_dict': 'efficientnet_b1-imagenet-7eb107fee3.pt'},
)
def efficientnet_b1(in_channels, num_classes):
    r"""EfficientNet B1

    See :class:`~ark.models.classification.efficientnet.EfficientNet` for details.
    """
    return EfficientNet(in_channels, num_classes,
                        depth_multiplier=1.1)


@register_model(
    imagenet1k={'in_channels': 3, 'num_classes': 1000,
                'state_dict': 'efficientnet_b2-imagenet-03de86b71c.pt'},
)
def efficientnet_b2(in_channels, num_classes):
    r"""EfficientNet B2

    See :class:`~ark.models.classification.efficientnet.EfficientNet` for details.
    """
    return EfficientNet(in_channels, num_classes,
                        width_multiplier=1.1,
                        depth_multiplier=1.2,
                        dropout_rate=0.3)


@register_model(
    imagenet1k={'in_channels': 3, 'num_classes': 1000,
                'state_dict': 'efficientnet_b3-imagenet-4105432b3b.pt'},
)
def efficientnet_b3(in_channels, num_classes):
    r"""EfficientNet B3

    See :class:`~ark.models.classification.efficientnet.EfficientNet` for details.
    """
    return EfficientNet(in_channels, num_classes,
                        width_multiplier=1.2,
                        depth_multiplier=1.4,
                        dropout_rate=0.3)


def efficientnet_b4(in_channels, num_classes):
    r"""EfficientNet B4

    See :class:`~ark.models.classification.efficientnet.EfficientNet` for details.
    """
    return EfficientNet(in_channels, num_classes,
                        width_multiplier=1.4,
                        depth_multiplier=1.8,
                        dropout_rate=0.4)


def efficientnet_b5(in_channels, num_classes):
    r"""EfficientNet B5

    See :class:`~ark.models.classification.efficientnet.EfficientNet` for details.
    """
    return EfficientNet(in_channels, num_classes,
                        width_multiplier=1.6,
                        depth_multiplier=2.2,
                        dropout_rate=0.4)


def efficientnet_b6(in_channels, num_classes):
    r"""EfficientNet B6

    See :class:`~ark.models.classification.efficientnet.EfficientNet` for details.
    """
    return EfficientNet(in_channels, num_classes,
                        width_multiplier=1.8,
                        depth_multiplier=2.6,
                        dropout_rate=0.5)


def efficientnet_b7(in_channels, num_classes):
    r"""EfficientNet B7

    See :class:`~ark.models.classification.efficientnet.EfficientNet` for details.
    """
    return EfficientNet(in_channels, num_classes,
                        width_multiplier=2.0,
                        depth_multiplier=3.1,
                        dropout_rate=0.5)


class EfficientNet(nn.Sequential):
    r"""EfficientNet implementation from 
    `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks":
    <https://arxiv.org/abs/1905.11946>`_ paper.

    Args:
        in_channels (int): the input channels
        num_classes (int): the number of the output classification classes
        width_multiplier (float): hyperparameter that scales the width 
            of the network
        depth_multiplier (float): hyperparameter that scales the depth of each 
            layer of the network
        dropout_p (float): probability for the dropout layer the before
            the last dense layer
    """

    def __init__(self, in_channels: int, num_classes: int,
                 width_multiplier: float = 1.,
                 depth_multiplier: float = 1.,
                 dropout_p: float = 0.2):

        def c(channels): return round_channels(width_multiplier * channels)
        def d(depth): return ceil(depth * depth_multiplier)

        def make_layer(in_channels, out_channels, num_blocks,
                       kernel_size=3, stride=1, expansion=6):
            MB = MobileInvertedBottleneck  # for redability
            layers = [
                MB(in_channels, out_channels, kernel_size,
                   stride=stride, expansion=expansion)
            ]
            for _ in range(1, num_blocks):
                layers += [
                    MB(out_channels, out_channels, kernel_size,
                       expansion=expansion)
                ]
            return nn.Sequential(*layers)

        features = nn.Sequential(OrderedDict([
            ('stem', ConvBnSwish2d(in_channels, c(32), 3, padding=1, stride=2)),
            ('layer1', make_layer(c(32), c(16), d(1), expansion=1)),
            ('layer2', make_layer(c(16), c(24), d(2), stride=2)),
            ('layer3', make_layer(c(24), c(40), d(2), kernel_size=5, stride=2)),
            ('layer4', make_layer(c(40), c(80), d(3), stride=2)),
            ('layer5', make_layer(c(80), c(112), d(3), kernel_size=5)),
            ('layer6', make_layer(c(112), c(192), d(4), kernel_size=5, stride=2)),
            ('layer7', make_layer(c(192), c(320), d(1))),
            ('tail', ConvBnSwish2d(c(320), c(1280), 1)),
        ]))

        classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Dropout(dropout_p),
            nn.Flatten(),
            nn.Linear(c(1280), num_classes),
        )

        super().__init__(OrderedDict([
            ('features', features),
            ('classifier', classifier),
        ]))


class MobileInvertedBottleneck(nn.Module):
    "MBBlock of EfficientNet"

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1,
                 expansion=6, reduction=4,
                 dropout_p=0.2):
        super(MobileInvertedBottleneck, self).__init__()

        hidden_channels = in_channels * expansion

        self.conv1 = (
            ConvBnSwish2d(in_channels, hidden_channels, 1)
            if expansion != 1
            else nn.Identity()
        )

        self.conv2 = ConvBnSwish2d(hidden_channels, hidden_channels, kernel_size,
                                   padding=kernel_size // 2,
                                   stride=stride,
                                   groups=hidden_channels)

        self.se = SqueezeExcitation(hidden_channels, hidden_channels,
                                    reduction=reduction*expansion)

        self.conv3 = ConvBn2d(hidden_channels, out_channels, 1)

        self.dropout = nn.Dropout2d(p=dropout_p)

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.se(x)
        x = self.conv3(x)
        if input.shape == x.shape:
            return input + self.dropout(x)
        else:
            return x


class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=4):
        super(SqueezeExcitation, self).__init__()

        red_channels = in_channels // reduction

        self.conv1 = nn.Conv2d(in_channels, red_channels, 1)
        self.act = Swish(inplace=True)
        self.conv2 = nn.Conv2d(red_channels, out_channels, 1)

    def forward(self, input):
        x = F.adaptive_avg_pool2d(input, 1)
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        return input * torch.sigmoid_(x)
