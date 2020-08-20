from typing import Tuple
from collections import OrderedDict
import torch
from torch import nn

from ark.nn.easy import ConvReLU2d
from ark.utils.hub import register_model

__all__ = [
    'SqueezeNet',
    'squeezenet',
]


@register_model(
    imagenet1k={'in_channels': 3, 'num_classes': 1000,
                'state_dict': 'https://files.deeplar.tk/ark-weights/squeezenet-imagenet-ed8e93b737.pt'},
)
def squeezenet(in_channels, num_classes):
    r"""SqueezeNet

    See :class:`~ark.models.classification.squeezenet.SqueezeNet` for details.
    """
    return SqueezeNet(in_channels, num_classes)


class SqueezeNet(nn.Sequential):
    r"""SqueezeNet implementation from the
    `"SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size":
    <https://arxiv.org/abs/1602.07360>` paper.

    Args:
        in_channels (int): the input channels
        num_classes (int): the number of the output classification classes
    """

    def __init__(self, in_channels: int, num_classes: int):
        features = nn.Sequential(OrderedDict([
            ('stem', nn.Sequential(
                ConvReLU2d(in_channels, 64, kernel_size=3, stride=2),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
            )),
            ('layer1', nn.Sequential(
                FireModule(64, 128, (16, 64, 64)),
                FireModule(128, 128, (16, 64, 64)),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            )),
            ('layer2', nn.Sequential(
                FireModule(128, 256, (32, 128, 128)),
                FireModule(256, 256, (32, 128, 128)),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            )),
            ('layer3', nn.Sequential(
                FireModule(256, 384, (48, 192, 192)),
                FireModule(384, 384, (48, 192, 192)),
                FireModule(384, 512, (64, 256, 256)),
                FireModule(512, 512, (64, 256, 256)),
            )),
        ]))

        classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(512, num_classes, 1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        super().__init__(OrderedDict([
            ('features', features),
            ('classifier', classifier),
        ]))


class FireModule(nn.Module):
    r"""Fire block of the SqueezeNet model.

    Args:
        in_channels (int): the input channels
        out_channels (int): the output channels
        layer_channels (tuple of int): the channels for the squeeze layer and 
            the 1x1 and 3x3 expand layer, respectively
    """

    def __init__(self, in_channels: int, out_channels: int,
                 layer_channels: Tuple[int, int, int]):
        super(FireModule, self).__init__()
        squeeze_channels, expand1x1_channels, expand3x3_channels = layer_channels

        assert expand1x1_channels + expand3x3_channels == out_channels, \
            'the output channels must match the sum of the expand channels'

        self.squeeze = ConvReLU2d(in_channels, squeeze_channels, 1)
        self.expand1x1 = ConvReLU2d(
            squeeze_channels, expand1x1_channels, 1)
        self.expand3x3 = ConvReLU2d(
            squeeze_channels, expand3x3_channels, 3, padding=1)

    def forward(self, input):
        x = self.squeeze(input)
        x1 = self.expand1x1(x)
        x2 = self.expand3x3(x)
        return torch.cat([x1, x2], dim=1)
