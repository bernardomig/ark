from collections import OrderedDict
import torch
from torch import nn

from ark.utils.hub import register_model


@register_model(
    cityscapes={'in_channels': 3, 'out_channels': 19,
                'state_dict': 'https://files.deeplar.tk/ark-weights/erfnet-cityscapes-ab3fd024d4.pt'},
)
def erfnet(in_channels, out_channels):
    return ERFNet(in_channels, out_channels)


class ERFNet(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(OrderedDict([
            ('layer1', nn.Sequential(
                DownsamplingBlock(3, 16)
            )),
            ('layer2', nn.Sequential(
                DownsamplingBlock(16, 64),
                NonBottleneck1dBlock(64, 64, dropout_p=0.03),
                NonBottleneck1dBlock(64, 64, dropout_p=0.03),
                NonBottleneck1dBlock(64, 64, dropout_p=0.03),
                NonBottleneck1dBlock(64, 64, dropout_p=0.03),
                NonBottleneck1dBlock(64, 64, dropout_p=0.03),
            )),
            ('layer3', nn.Sequential(
                DownsamplingBlock(64, 128),
                NonBottleneck1dBlock(128, 128, dilation=2, dropout_p=0.3),
                NonBottleneck1dBlock(128, 128, dilation=4, dropout_p=0.3),
                NonBottleneck1dBlock(128, 128, dilation=8, dropout_p=0.3),
                NonBottleneck1dBlock(128, 128, dilation=16, dropout_p=0.3),
                NonBottleneck1dBlock(128, 128, dilation=2, dropout_p=0.3),
                NonBottleneck1dBlock(128, 128, dilation=4, dropout_p=0.3),
                NonBottleneck1dBlock(128, 128, dilation=8, dropout_p=0.3),
                NonBottleneck1dBlock(128, 128, dilation=16, dropout_p=0.3),
            )),
            ('layer4', nn.Sequential(
                UpsamplingBlock(128, 64),
                NonBottleneck1dBlock(64, 64, dropout_p=0),
                NonBottleneck1dBlock(64, 64, dropout_p=0),
            )),
            ('layer5', nn.Sequential(
                UpsamplingBlock(64, 16),
                NonBottleneck1dBlock(16, 16, dropout_p=0),
                NonBottleneck1dBlock(16, 16, dropout_p=0),
            )),
            ('classifier', nn.ConvTranspose2d(16, out_channels, 2, stride=2, padding=0, output_padding=0)),
        ]))


class DownsamplingBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels - in_channels, 3, padding=1, stride=2)
        self.pool = nn.MaxPool2d(2)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        x = torch.cat([
            self.conv(input), self.pool(input)
        ], dim=1)
        x = self.bn(x)
        return torch.relu_(x)


class UpsamplingBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(OrderedDict([
            ('conv', nn.ConvTranspose2d(in_channels, out_channels, 3, padding=1, output_padding=1, stride=2)),
            ('bn', nn.BatchNorm2d(out_channels)),
            ('relu', nn.ReLU(inplace=True)),
        ]))


class NonBottleneck1dBlock(nn.Module):

    def __init__(self, in_channels, out_channels, dilation=1, dropout_p=0.1):
        super().__init__()

        assert in_channels == out_channels

        channels = in_channels

        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 1), padding=(1, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, (1, 3), padding=(0, 1)),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 1), padding=(dilation, 0), dilation=(dilation, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, (1, 3), padding=(0, dilation), dilation=(1, dilation)),
            nn.BatchNorm2d(channels),
        )

        self.activation = nn.ReLU(inplace=True)

        self.dropout = nn.Dropout2d(dropout_p)

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.dropout(x)
        return self.activation(x + input)
