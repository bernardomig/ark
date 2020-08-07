from typing import Tuple
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F

from ark.nn.easy import ConvBn2d


def enet(in_channels, out_channels):
    return ENet(in_channels, out_channels)


class ENet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ENet, self).__init__()

        self.head = InitialBlock(in_channels, 16)

        self.layer1 = nn.ModuleList([
            DownsamplingBottleneck(16, 64, dropout_p=0.01),
            RegularBottleneck(64, 64, dropout_p=0.01),
            RegularBottleneck(64, 64, dropout_p=0.01),
            RegularBottleneck(64, 64, dropout_p=0.01),
            RegularBottleneck(64, 64, dropout_p=0.01),
        ])

        self.layer2 = nn.ModuleList([
            DownsamplingBottleneck(64, 128),
            RegularBottleneck(128, 128, dropout_p=0.1),
            RegularBottleneck(128, 128, dilation=2, dropout_p=0.1),
            RegularBottleneck(128, 128, kernel_size=5, dropout_p=0.1),
            RegularBottleneck(128, 128, dilation=4, dropout_p=0.1),
            RegularBottleneck(128, 128, dropout_p=0.1),
            RegularBottleneck(128, 128, dilation=8, dropout_p=0.1),
            RegularBottleneck(128, 128, kernel_size=5, dropout_p=0.1),
            RegularBottleneck(128, 128, dilation=16, dropout_p=0.1),
        ])

        self.layer3 = nn.ModuleList([
            RegularBottleneck(128, 128),
            RegularBottleneck(128, 128, dilation=2, dropout_p=0.1),
            RegularBottleneck(128, 128, kernel_size=5, dropout_p=0.1),
            RegularBottleneck(128, 128, dilation=4, dropout_p=0.1),
            RegularBottleneck(128, 128),
            RegularBottleneck(128, 128, dilation=8, dropout_p=0.1),
            RegularBottleneck(128, 128, kernel_size=5, dropout_p=0.1),
            RegularBottleneck(128, 128, dilation=16, dropout_p=0.1),
        ])

        self.layer4 = nn.ModuleList([
            UpsamplingBottleneck(128, 64),
            RegularBottleneck(64, 64, dropout_p=0.1),
            RegularBottleneck(64, 64, dropout_p=0.1),
        ])

        self.layer5 = nn.ModuleList([
            UpsamplingBottleneck(64, 16),
            RegularBottleneck(16, 16, dropout_p=0.1),
        ])

        self.classifier = nn.Conv2d(16, out_channels, 1)

    def forward(self, input):
        x = self.head(input)
        indices = []
        layers = [self.layer1, self.layer2, self.layer3,
                  self.layer4, self.layer5]
        for layer in layers:
            for block in layer.children():
                if isinstance(block, RegularBottleneck):
                    x = block(x)
                elif isinstance(block, DownsamplingBottleneck):
                    x, i = block(x)
                    indices = [i, *indices]
                elif isinstance(block, UpsamplingBottleneck):
                    i, *indices = indices
                    x = block(x, i)
                else:
                    raise RuntimeError("Unknown block type in ENet")

        x = self.classifier(x)

        return F.interpolate(x, scale_factor=2,
                             mode='bilinear', align_corners=True)


class InitialBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        assert out_channels > in_channels, \
            "output channels must be greater than the input channels"

        self.conv = ConvBn2d(in_channels, out_channels - in_channels, 3,
                             padding=1, stride=2)
        self.pool = nn.MaxPool2d(2)
        self.activation = nn.PReLU()


class RegularBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, projection_ratio=4, dropout_p=0.1):
        super(RegularBottleneck, self).__init__()

        assert kernel_size in {3, 5}

        width = in_channels // projection_ratio
        self.conv1 = ConvBnPReLU2d(in_channels, width, 1)
        self.conv2 = (
            ConvBnPReLU2d(width, width, 3, padding=dilation, dilation=dilation)
            if kernel_size == 3
            else nn.Sequential(
                ConvBn2d(width, width, (1, 5), padding=(0, 2)),
                ConvBn2d(width, width, (5, 1), padding=(2, 0)),
                nn.PReLU(),
            )
        )
        self.conv3 = ConvBn2d(width, out_channels, 1)

        self.activation = nn.PReLU()

        self.dropout = nn.Dropout2d(p=dropout_p)

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.dropout(x)
        residual = pad_zeros(input, x.shape)
        return self.activation(x + residual)


class DownsamplingBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels,
                 projection_ratio=4, dropout_p=0.1):
        super(DownsamplingBottleneck, self).__init__()

        width = in_channels // projection_ratio

        self.conv1 = ConvBnPReLU2d(in_channels, width, 1)
        self.conv2 = ConvBnPReLU2d(width, width, 3, 1, stride=2)
        self.conv3 = ConvBn2d(width, out_channels, 1)
        self.dropout = nn.Dropout2d(p=dropout_p)

        self.downsample = nn.MaxPool2d(kernel_size=2, return_indices=True)

        self.activation = nn.PReLU()

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.dropout(x)

        residual, indices = self.downsample(input)
        residual = pad_zeros(residual, x.shape)

        return self.activation(x + residual), indices


class UpsamplingBottleneck(nn.Module):

    def __init__(self, in_channels, out_channels,
                 projection_ratio=4, dropout_p=0.1):
        super(UpsamplingBottleneck, self).__init__()

        width = in_channels // projection_ratio

        self.conv1 = ConvBnPReLU2d(in_channels, width, 1)
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(width, width, 3,
                               stride=2, padding=1,
                               output_padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.PReLU(),
        )

        # TODO: self.conv3 should be a ConvBn2d (no activation!)
        #       retraining a new model is required
        self.conv3 = ConvBnPReLU2d(width, out_channels, 1)
        self.dropout = nn.Dropout2d(p=dropout_p)

        self.upsample = nn.ModuleDict({
            'unpool': nn.MaxUnpool2d(kernel_size=2, stride=2),
            'conv': ConvBn2d(in_channels, out_channels, 1),
        })

        self.activation = nn.PReLU()

    def forward(self, input, indices):
        left = self.conv1(input)
        left = self.conv2(left)
        left = self.conv3(left)
        left = self.dropout(left)

        right = self.upsample['conv'](input)
        right = self.upsample['unpool'](
            right, indices=indices, output_size=left.size())

        return self.activation(left + right)


class ConvBnPReLU2d(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, dilation=1):
        super().__init__(OrderedDict([
            ('conv', nn.Conv2d(in_channels, out_channels, kernel_size,
                               padding=padding, stride=stride, dilation=dilation,
                               bias=False)),
            ('bn', nn.BatchNorm2d(out_channels)),
            ('relu', nn.PReLU()),
        ]))


@torch.jit.script
def pad_zeros(input: torch.Tensor, shape: Tuple[int, int, int, int]):
    if input.shape == shape:
        return input

    b, _, h, w = input.shape

    c = int(shape[1] - input.size(1))

    pad = torch.zeros((b, c, h, w)).to(input)
    return torch.cat([input, pad], dim=1)
