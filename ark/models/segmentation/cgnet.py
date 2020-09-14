from collections import OrderedDict
import torch
from torch import nn
from torch.nn import functional as F


def cgnet(in_channels, out_channels):
    return CGNet(in_channels, out_channels)


class CGNet(nn.Module):
    def __init__(self, in_channels, out_channels, depth=[3, 21]):
        super().__init__()

        self.stage1 = nn.Sequential(
            ConvBnPReLU2d(in_channels, 32, 3, padding=1, stride=2),
            ConvBnPReLU2d(32, 32, 3, padding=1),
            ConvBnPReLU2d(32, 32, 3, padding=1),
        )

        self.stage2 = nn.ModuleList(
            CGBlock(32 + in_channels if i == 0 else 64, 64,
                    stride=2 if i == 0 else 1,
                    dilation=2, reduction=8)
            for i in range(depth[0])
        )

        self.stage3 = nn.ModuleList(
            CGBlock(128 + in_channels if i == 0 else 128, 128,
                    stride=2 if i == 0 else 1,
                    dilation=4, reduction=16)
            for i in range(depth[1])
        )

        self.classifier = nn.Sequential(
            nn.Dropout2d(p=0.1),
            nn.Conv2d(256, out_channels, 1)
        )

    def forward(self, input):
        half = F.avg_pool2d(input, 3, stride=2, padding=1)
        quarter = F.avg_pool2d(half, 3, stride=2, padding=1)

        x = self.stage1(input)
        x = torch.cat([x, half], dim=1)
        x = x2 = self.stage2[0](x)
        for layer in self.stage2[1:]:
            x = layer(x)
        x = torch.cat([x, x2, quarter], dim=1)
        x = x3 = self.stage3[0](x)
        for layer in self.stage3[1:]:
            x = layer(x)
        x = torch.cat([x, x3], dim=1)
        x = self.classifier(x)
        return F.interpolate(x, size=input.shape[2:], mode='bilinear', align_corners=True)


class CGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=2, reduction=16):
        super().__init__()

        width = out_channels // 2 if stride == 1 else out_channels

        self.conv1 = (
            ConvBnPReLU2d(in_channels, width, 1)
            if stride == 1
            else ConvBnPReLU2d(in_channels, width, 3, padding=1, stride=stride)
        )

        self.conv2 = nn.ModuleDict({
            'conv': nn.ModuleDict({
                'loc': nn.Conv2d(width, width, 3, padding=1, groups=width, bias=False),
                'sur': nn.Conv2d(width, width, 3, padding=dilation, dilation=dilation, groups=width, bias=False),
            }),
            'bn': nn.BatchNorm2d(width * 2),
            'relu': nn.PReLU(width * 2),
        })

        self.conv3 = (
            nn.Conv2d(width * 2, out_channels, 1, bias=False)
            if stride != 1
            else nn.Identity()
        )

        self.se = SqueezeExcitation(out_channels, out_channels, reduction=reduction)

    def forward(self, input):
        x = self.conv1(input)
        x = torch.cat([
            self.conv2['conv']['loc'](x),
            self.conv2['conv']['sur'](x),
        ], dim=1)
        x = self.conv2['bn'](x)
        x = self.conv2['relu'](x)
        x = self.conv3(x)
        x = self.se(x)

        if x.shape == input.shape:
            x = x + input
        return x


class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=4):
        super(SqueezeExcitation, self).__init__()

        red_channels = in_channels // reduction

        self.conv1 = nn.Conv2d(in_channels, red_channels, 1)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(red_channels, out_channels, 1)

    def forward(self, input):
        x = F.adaptive_avg_pool2d(input, 1)
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        return input * torch.sigmoid_(x)


class ConvBnPReLU2d(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, dilation=1):
        super().__init__(OrderedDict([
            ('conv', nn.Conv2d(in_channels, out_channels, kernel_size,
                               padding=padding, stride=stride, dilation=dilation,
                               bias=False)),
            ('bn', nn.BatchNorm2d(out_channels)),
            ('relu', nn.PReLU(out_channels)),
        ]))
