from typing import Tuple
from collections import OrderedDict
import torch
from torch import nn

from ark.nn.easy import ConvBnReLU2d

__all__ = [
    'DenseNet',
    'densenet121', 'densenet161',
    'densenet169', 'densenet201'
]


def densenet121(in_channels, num_classes):
    r"""DenseNet121 model

    See :class:`~ark.models.classification.densenet.DenseNet` for details.
    """
    return DenseNet(in_channels, num_classes,
                    growth=32,
                    block_depth=[6, 12, 24, 16],
                    init_channels=64)


def densenet161(in_channels, num_classes):
    r"""DenseNet161 model

    See :class:`~ark.models.classification.densenet.DenseNet` for details.
    """
    return DenseNet(in_channels, num_classes,
                    growth=48,
                    block_depth=[6, 12, 36, 24],
                    init_channels=96)


def densenet169(in_channels, num_classes):
    r"""DenseNet169 model

    See :class:`~ark.models.classification.densenet.DenseNet` for details.
    """
    return DenseNet(in_channels, num_classes,
                    growth=32,
                    block_depth=[6, 12, 32, 32],
                    init_channels=64)


def densenet201(in_channels, num_classes):
    r"""DenseNet201 model

    See :class:`~ark.models.classification.densenet.DenseNet` for details.
    """
    return DenseNet(in_channels, num_classes,
                    growth=32,
                    block_depth=[6, 12, 48, 32],
                    init_channels=64)


class DenseNet(nn.Sequential):
    r"""DenseNet implementation from
    `"Densely Connected Convolutional Networks": 
    <https://arxiv.org/abs/1608.06993>`_ paper.

    Args:
        in_channels (int): the input channels
        num_classes (int): the number of the output classification classes
        growth (int): hyperparameter that determines the growth of each dense block
        block_depth (tuple of int): the number of blocks in each layer
        init_channels (int): the number of output features of the stem block
        expansion (int): expansion at each dense block
        dropout_p (float): the dropout ratio at each dense block
    """

    def __init__(self,
                 in_channels: int,
                 num_classes: int,
                 growth: int,
                 block_depth: Tuple[int, int, int, int],
                 init_channels: int,
                 expansion: int = 4,
                 dropout_p: float = 0.0):

        layers = OrderedDict()

        layers["stem"] = nn.Sequential(OrderedDict([
            ('conv', ConvBnReLU2d(in_channels, init_channels, 7, padding=3, stride=2)),
            ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        channels = init_channels
        for idx, depth in enumerate(block_depth):
            out_channels = channels + growth * depth
            layer = [
                DenseBlock(channels + growth * idx, channels + growth * (idx+1),
                           growth=growth,
                           expansion=expansion,
                           dropout_p=dropout_p)
                for idx in range(depth)
            ]
            if idx != len(block_depth) - 1:
                # densenet does not have a transition block in the last layer
                layer += TransitionBlock(out_channels, out_channels // 2)
                out_channels = out_channels // 2

            layers[f"layer{idx+1}"] = nn.Sequential(*layer)
            channels = out_channels

        layers['tail'] = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        features = nn.Sequential(layers)

        classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, num_classes)
        )

        super(DenseNet, self).__init__(OrderedDict([
            ('features', features),
            ('classifier', classifier),
        ]))


class DenseBlock(nn.Module):
    r"""The Densenet building block, contructed by two pre-activated convolutions.

    In this implementation, we chose to concatenate the output of the last conv layer
    with the input, so that we can construct each densenet layer using a sequencial
    model. Therefore, the number of output channels is constrained by the input channels
    and the growth hyperparameter.

    Args:
        in_channels (int): the input channels
        out_channels (int): the output channels. Should be equal to the 
            input channels plus the growth.
        growth (int): the block growth hyperparameter
        expansion (int): defines the number of parameters in the middle 
            of the two conv blocks
        dropout_p (float): the dropout probability before the feature 
            concatenation
    """

    def __init__(self, in_channels, out_channels, growth, expansion, dropout_p):
        super(DenseBlock, self).__init__()

        assert in_channels + growth == out_channels

        self.conv1 = BnReLUConv2d(in_channels, growth * expansion, 1)
        self.conv2 = BnReLUConv2d(growth * expansion, growth, 3, padding=1)

        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.dropout(x)
        return torch.cat([input, x], dim=1)


class TransitionBlock(nn.Sequential):
    r"""The DenseNet transition block, constructed by a pre-activated conv 
    followed by a average pooling operation.
    """

    def __init__(self, in_channels, out_channels):
        super(TransitionBlock, self).__init__(OrderedDict([
            ('conv', BnReLUConv2d(in_channels, out_channels, 1)),
            ('pool', nn.AvgPool2d(2)),
        ]))


class BnReLUConv2d(nn.Sequential):
    r"""A sequence of batchnorm+relu+conv"""

    def __init__(self, in_channels, out_channels, kernel_size,
                 padding=0,
                 stride=1,
                 dilation=1,
                 groups=1):
        super(BnReLUConv2d, self).__init__(OrderedDict([
            ('bn', nn.BatchNorm2d(in_channels)),
            ('relu', nn.ReLU(inplace=True)),
            ('conv', nn.Conv2d(in_channels, out_channels, kernel_size,
                               padding=padding,
                               stride=stride,
                               dilation=dilation,
                               groups=groups,
                               bias=False)),
        ]))
