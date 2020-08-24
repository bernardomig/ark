from collections import OrderedDict
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F

from ark.nn.easy import ConvBnReLU2d, ConvBn2d
from ark.nn.functional import channel_shuffle
from ark.utils.hub import register_model

__all__ = [
    'LedNet',
    'lednet',
]


@register_model(
    cityscapes={'in_channels': 3, 'out_channels': 19,
                'state_dict': 'https://files.deeplar.tk/ark-weights/lednet-cityscapes-9fa2a8ee12.pt'},
)
def lednet(in_channels, out_channels):
    return LedNet(in_channels, out_channels)


class LedNet(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(LedNet, self).__init__()

        self.encoder = nn.Sequential(OrderedDict([
            ('layer1', nn.Sequential(
                DownsamplingBlock(in_channels, 32),
                SSnbtBlock(32, 32, dropout_p=0.03),
                SSnbtBlock(32, 32, dropout_p=0.03),
                SSnbtBlock(32, 32, dropout_p=0.03),
            )),
            ('layer2', nn.Sequential(
                DownsamplingBlock(32, 64),
                SSnbtBlock(64, 64, dropout_p=0.03),
                SSnbtBlock(64, 64, dropout_p=0.03),
            )),
            ('layer3', nn.Sequential(
                DownsamplingBlock(64, 128),
                SSnbtBlock(128, 128),
                SSnbtBlock(128, 128, dilation=2, dropout_p=0.3),
                SSnbtBlock(128, 128, dilation=5, dropout_p=0.3),
                SSnbtBlock(128, 128, dilation=9, dropout_p=0.3),
                SSnbtBlock(128, 128, dilation=2, dropout_p=0.3),
                SSnbtBlock(128, 128, dilation=5, dropout_p=0.3),
                SSnbtBlock(128, 128, dilation=9, dropout_p=0.3),
                SSnbtBlock(128, 128, dilation=17, dropout_p=0.3),
            )),
        ]))

        self.decoder = APNModule(128, out_channels)

    def forward(self, input):
        x = self.encoder(input)
        x = self.decoder(x)
        x = F.interpolate(
            x, scale_factor=8,
            mode='bilinear', align_corners=True)
        return x


class APNModule(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(APNModule, self).__init__()

        self.conv1 = ConvBnReLU2d(in_channels, in_channels, 3, 1, stride=2)
        self.conv2 = ConvBnReLU2d(in_channels, in_channels, 5, 2, stride=2)
        self.conv3 = ConvBnReLU2d(in_channels, in_channels, 7, 3, stride=2)

        self.level1 = ConvBnReLU2d(in_channels, out_channels, 1)
        self.level2 = ConvBnReLU2d(in_channels, out_channels, 1)
        self.level3 = ConvBnReLU2d(in_channels, out_channels, 1)
        self.level4 = ConvBnReLU2d(in_channels, out_channels, 1)
        self.level5 = ConvBnReLU2d(in_channels, out_channels, 1)

    def forward(self, input):
        scale = partial(F.interpolate, scale_factor=2,
                        mode='bilinear', align_corners=True)

        b3 = self.conv1(input)
        b2 = self.conv2(b3)
        b1 = self.conv3(b2)

        b1 = self.level1(b1)
        b2 = self.level2(b2)
        b3 = self.level3(b3)

        x = scale(b3 + scale(b2 + scale(b1)))

        x = x * self.level4(input)

        p = F.adaptive_avg_pool2d(input, 1)
        x = x + self.level5(p)

        return x


class SSnbtBlock(nn.Module):

    def __init__(self, in_channels, out_channels, dilation=1, dropout_p=0.0):
        super(SSnbtBlock, self).__init__()

        if in_channels != out_channels:
            raise ValueError("input and output channels must match")

        channels = in_channels // 2
        self.left = nn.Sequential(
            ConvBnReLU2d(channels, channels, (3, 1), padding=(1, 0)),
            ConvBnReLU2d(channels, channels, (1, 3), padding=(0, 1)),
            ConvBnReLU2d(channels, channels, (3, 1),
                         padding=(dilation, 0), dilation=(dilation, 1)),
            ConvBn2d(channels, channels, (1, 3), padding=(0, dilation),
                     dilation=(1, dilation)),
        )
        self.right = nn.Sequential(
            ConvBnReLU2d(channels, channels, (3, 1), padding=(1, 0)),
            ConvBnReLU2d(channels, channels, (1, 3), padding=(0, 1)),
            ConvBnReLU2d(channels, channels, (3, 1),
                         padding=(dilation, 0), dilation=(dilation, 1)),
            ConvBn2d(channels, channels, (1, 3), padding=(0, dilation),
                     dilation=(1, dilation)),
        )

        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=dropout_p)

    def forward(self, input):
        left, right = torch.chunk(input, 2, 1)
        left = self.left(left)
        right = self.right(right)
        x = torch.cat([left, right], dim=1)
        x = self.dropout(x)
        x = self.activation(input + x)
        return channel_shuffle(x, 2)


class DownsamplingBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DownsamplingBlock, self).__init__()

        self.conv = ConvBn2d(in_channels, out_channels - in_channels,
                             kernel_size=3, padding=1, stride=2)
        self.pool = nn.MaxPool2d(kernel_size=2, ceil_mode=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        x = torch.cat([
            self.conv(input),
            self.pool(input),
        ], dim=1)
        x = self.relu(x)
        return x
