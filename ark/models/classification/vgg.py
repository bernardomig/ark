from typing import Tuple
from functools import partial
from collections import OrderedDict

from torch import nn

from ark.nn.easy import ConvBnReLU2d


def vgg11(in_channels, num_classes):
    r"""VGG11

    See :class:`~ark.models.classification.vgg.VGG` for details.
    """
    return VGG(in_channels, num_classes, block_depth=[1, 1, 2, 2, 2])


def vgg13(in_channels, num_classes):
    r"""VGG13

    See :class:`~ark.models.classification.vgg.VGG` for details.
    """
    return VGG(in_channels, num_classes, block_depth=[2, 2, 2, 2, 2])


def vgg16(in_channels, num_classes):
    r"""VGG16

    See :class:`~ark.models.classification.vgg.VGG` for details.
    """
    return VGG(in_channels, num_classes, block_depth=[2, 2, 3, 3, 3])


def vgg19(in_channels, num_classes):
    r"""VGG19

    See :class:`~ark.models.classification.vgg.VGG` for details.
    """
    return VGG(in_channels, num_classes, block_depth=[2, 2, 4, 4, 4])


class VGG(nn.Sequential):
    r"""VGG implementation from the 
    `"Very Deep Convolutional Networks for Large-Scale Image Recognition": 
    <https://arxiv.org/abs/1409.1556>` paper.

    This version includes Batch Normalization for better accuracy and easier 
    training.

    Args:
        in_channels (int): the input channels
        num_classes (int): the number of the output classification classes
        block_depth (tuple of int): number of convs per layer
    """

    def __init__(self, in_channels: int, num_classes: int,
                 block_depth: Tuple[int, int, int, int]):

        def make_layer(in_channels, out_channels, num_blocks):
            ConvBlock = partial(ConvBnReLU2d, kernel_size=3, padding=1)

            layers = [ConvBlock(in_channels, out_channels)]
            for _ in range(1, num_blocks):
                layers += [ConvBlock(out_channels, out_channels)]
            layers += [nn.MaxPool2d(kernel_size=2)]
            return nn.Sequential(*layers)

        features = nn.Sequential(OrderedDict([
            ('layer1', make_layer(in_channels, 64, block_depth[0])),
            ('layer2', make_layer(64, 128, block_depth[1])),
            ('layer3', make_layer(128, 256, block_depth[2])),
            ('layer4', make_layer(256, 512, block_depth[3])),
            ('layer5', make_layer(512, 512, block_depth[4])),
        ]))

        classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7 * 7 * 512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

        super().__init__(OrderedDict([
            ('features', features),
            ('classifier', classifier),
        ]))
