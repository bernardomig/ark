from typing import Tuple
from collections import OrderedDict
import torch
from torch import nn


def preact_resnet18(in_channels, num_classes):
    r"""Pre-activation ResNet 18

    See :class:`~ark.models.classification.preact_resnet.PreactResNet` for details.
    """
    return PreactResNet(in_channels, num_classes,
                        block=BasicBlock,
                        block_depth=[2, 2, 2, 2],
                        init_channels=64,
                        block_channels=[64, 128, 256, 512])


def preact_resnet34(in_channels, num_classes):
    r"""Pre-activation ResNet 34

    See :class:`~ark.models.classification.preact_resnet.PreactResNet` for details.
    """
    return PreactResNet(in_channels, num_classes,
                        block=BasicBlock,
                        block_depth=[3, 4, 6, 3],
                        init_channels=64,
                        block_channels=[64, 128, 256, 512])


def preact_resnet50(in_channels, num_classes):
    r"""Pre-activation ResNet 50

    See :class:`~ark.models.classification.preact_resnet.PreactResNet` for details.
    """
    return PreactResNet(in_channels, num_classes,
                        block=Bottleneck,
                        block_depth=[3, 4, 6, 3],
                        init_channels=64,
                        block_channels=[256, 512, 1024, 2048])


def preact_resnet101(in_channels, num_classes):
    r"""Pre-activation ResNet 101

    See :class:`~ark.models.classification.preact_resnet.PreactResNet` for details.
    """
    return PreactResNet(in_channels, num_classes,
                        block=Bottleneck,
                        block_depth=[3, 4, 23, 3],
                        init_channels=64,
                        block_channels=[256, 512, 1024, 2048])


def preact_resnet152(in_channels, num_classes):
    r"""Pre-activation ResNet 152

    See :class:`~ark.models.classification.preact_resnet.PreactResNet` for details.
    """
    return PreactResNet(in_channels, num_classes,
                        block=Bottleneck,
                        block_depth=[3, 8, 36, 3],
                        init_channels=64,
                        block_channels=[256, 512, 1024, 2048])


class PreactResNet(nn.Sequential):
    r"""Pre-activated ResNets implemented from the 
    `"Identity Mappings in Deep Residual Networks": 
    <https://arxiv.org/abs/1603.05027>` paper.

    Args:
        in_channels (int): the input channels
        num_classes (int): the number of the output classification classes
        block (:obj:~nn.Module): the residual block class
        block_depth (tuple of int): number of blocks per layer
        init_channels (int): the input channels of the feature encoder
        block_channels (tuple of int): number of channels per layer
    """

    def __init__(self, in_channels: int, num_classes: int,
                 block: nn.Module,
                 block_depth: Tuple[int, int, int, int],
                 init_channels: int,
                 block_channels: Tuple[int, int, int, int]):

        def make_layer(in_channels, out_channels, num_blocks, stride=1):
            layers = [block(in_channels, out_channels, stride=stride)]
            for _ in range(1, num_blocks):
                layers += [block(out_channels, out_channels)]
            return nn.Sequential(*layers)

        # for redability
        d = block_depth
        c = block_channels

        features = nn.Sequential(OrderedDict([
            ('stem', nn.Sequential(
                nn.Conv2d(in_channels, init_channels, 7, padding=2, stride=2, bias=False),
                nn.BatchNorm2d(init_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )),
            ('layer1', make_layer(init_channels, c[0], d[0], stride=1)),
            ('layer2', make_layer(c[0], c[1], d[1], stride=2)),
            ('layer3', make_layer(c[1], c[2], d[2], stride=2)),
            ('layer4', make_layer(c[2], c[3], d[3], stride=2)),
        ]))

        classifier = nn.Sequential(
            nn.BatchNorm2d(c[3]),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(c[3], num_classes),
        )

        super(PreactResNet, self).__init__(OrderedDict([
            ('features', features),
            ('classifier', classifier),
        ]))


class BasicBlock(nn.Module):
    "Pre-activated Basic residual block for the Preact ResNets"

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=stride, bias=False)

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)

        self.downsample = (
            nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)
            if stride != 1 or in_channels != out_channels
            else None
        )

    def forward(self, input):
        residual = input

        x = self.bn1(input)
        x = self.relu1(x)
        if self.downsample is not None:
            residual = self.downsample(input)
        x = self.conv1(x)

        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv2(x)

        return x + residual


class Bottleneck(nn.Module):
    "Pre-activated Bottleneck residual block for the Preact ResNets"

    def __init__(self, in_channels, out_channels,
                 stride=1, expansion=4):
        super().__init__()

        width = out_channels // expansion

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, width, 1, bias=False)

        self.bn2 = nn.BatchNorm2d(width)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width, width, 3, padding=1, stride=stride, bias=False)

        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width, out_channels, 1, bias=False)

        self.downsample = (
            nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)
            if stride != 1 or in_channels != out_channels
            else None
        )

    def forward(self, input):
        residual = input

        x = self.bn1(input)
        x = self.relu1(x)
        if self.downsample is not None:
            residual = x
        x = self.conv1(x)

        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv2(x)

        x = self.bn3(x)
        x = self.relu3(x)
        x = self.conv3(x)

        return x + residual
