from collections import OrderedDict
import torch
from torch import nn
from torch.nn import functional as F

from ark.nn.easy import ConvReLU2d, ConvBn2d, ConvBnReLU2d


def esnet(in_channels, out_channels):
    return ESNet(in_channels, out_channels)


class ESNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encoder = nn.Sequential(OrderedDict([
            ("layer1", nn.Sequential(
                Downsampling(in_channels, 16),
                FCU(16, 16, 3, dropout_p=0.03),
                FCU(16, 16, 3, dropout_p=0.03),
                FCU(16, 16, 3, dropout_p=0.03),
            )),
            ("layer2", nn.Sequential(
                Downsampling(16, 64),
                FCU(64, 64, 5, dropout_p=0.03),
                FCU(64, 64, 5, dropout_p=0.03),
            )),
            ("layer3", nn.Sequential(
                Downsampling(64, 128),
                PFCU(128, 128, dropout_p=0.3),
                PFCU(128, 128, dropout_p=0.3),
                PFCU(128, 128, dropout_p=0.3),
                PFCU(128, 128, dropout_p=0.3),
                PFCU(128, 128, dropout_p=0.3),
            )),
        ]))

        self.decoder = nn.Sequential(OrderedDict([
            ("layer1", nn.Sequential(
                Upsampling(128, 64),
                FCU(64, 64, 5, dropout_p=0),
                FCU(64, 64, 5, dropout_p=0),
            )),
            ("layer2", nn.Sequential(
                Upsampling(64, 16),
                FCU(16, 16, 3, dropout_p=0),
                FCU(16, 16, 3, dropout_p=0),
            ))
        ]))

        self.classifier = nn.Conv2d(16, out_channels, 1)

    def forward(self, input):
        x = self.encoder(input)
        x = self.decoder(x)
        x = self.classifier(x)
        return F.interpolate(x, size=input.shape[2:], mode='bilinear', align_corners=True)


class FCU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dropout_p=0.01):
        super().__init__()

        self.conv1 = nn.Sequential(
            ConvBnReLU2d(in_channels, out_channels, (kernel_size, 1), padding=(kernel_size//2, 0)),
            ConvBnReLU2d(out_channels, out_channels, (1, kernel_size), padding=(0, kernel_size//2)),
        )

        self.conv2 = nn.Sequential(
            ConvBnReLU2d(out_channels, out_channels, (kernel_size, 1), padding=(kernel_size//2, 0)),
            ConvBn2d(out_channels, out_channels, (1, kernel_size), padding=(0, kernel_size//2)),
        )

        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=dropout_p)

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.dropout(x)
        return self.activation(x + input)


class PFCU(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rates=(2, 5, 9), dropout_p=0.01):
        super().__init__()

        self.conv1 = nn.Sequential(
            ConvBnReLU2d(in_channels, out_channels, (3, 1), padding=(1, 0)),
            ConvBnReLU2d(out_channels, out_channels, (1, 3), padding=(0, 1)),
        )

        self.conv2 = nn.ModuleList(
            nn.Sequential(
                ConvBnReLU2d(out_channels, out_channels, (3, 1), padding=(dilation, 0), dilation=(dilation, 1)),
                ConvBn2d(out_channels, out_channels, (1, 3), padding=(0, dilation), dilation=(1, dilation)),
            )
            for dilation in dilation_rates
        )

        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=dropout_p)

    def forward(self, input):
        x = self.conv1(input)
        x = sum((conv(x) for conv in self.conv2))
        x = self.dropout(x)
        return self.activation(x + input)


class Downsampling(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Downsampling, self).__init__()

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


class Upsampling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(OrderedDict([
            ('conv', nn.ConvTranspose2d(in_channels, out_channels, 3, stride=2,
                                        padding=1, output_padding=1, bias=False)),
            ('bn', nn.BatchNorm2d(out_channels)),
            ('relu', nn.ReLU(inplace=True)),
        ]))
