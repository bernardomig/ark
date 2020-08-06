from collections import OrderedDict
import torch
from torch import nn

from ark.nn.easy import ConvReLU2d

__all__ = [
    'SqueezeNet',
    'squeezenet',
]


def squeezenet(in_channels, num_classes):
    return SqueezeNet(in_channels, num_classes)


class SqueezeNet(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        features = nn.Sequential(OrderedDict([
            ('stem', nn.Sequential(
                ConvReLU2d(in_channels, 64, kernel_size=3, stride=2),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
            )),
            ('layer1', nn.Sequential(
                FireBlock(64, 128, (16, 64, 64)),
                FireBlock(128, 128, (16, 64, 64)),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            )),
            ('layer2', nn.Sequential(
                FireBlock(128, 256, (32, 128, 128)),
                FireBlock(256, 256, (32, 128, 128)),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            )),
            ('layer3', nn.Sequential(
                FireBlock(256, 384, (48, 192, 192)),
                FireBlock(384, 384, (48, 192, 192)),
                FireBlock(384, 512, (64, 256, 256)),
                FireBlock(512, 512, (64, 256, 256)),
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


class FireBlock(nn.Module):
    def __init__(self, in_channels, out_channels, layer_channels):
        super().__init__()
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
