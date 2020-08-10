from collections import OrderedDict
from functools import partial
import torch
from torch import nn
from torch.nn import functional as F

from ark.nn.easy import ConvBnAct2d, ConvBn2d
from ark.nn.utils import round_channels


def mobilenetv3_small_1_0(in_channels, num_classes):
    return MobileNetV3Small(in_channels, num_classes)


def mobilenetv3_small_0_75(in_channels, num_classes):
    return MobileNetV3Small(in_channels, num_classes, width_multiplier=0.75)


def mobilenetv3_large_1_0(in_channels, num_classes):
    return MobileNetV3Large(in_channels, num_classes)


def mobilenetv3_large_0_75(in_channels, num_classes):
    return MobileNetV3Large(in_channels, num_classes, width_multiplier=0.75)


class MobileNetV3(nn.Sequential):
    def __init__(self, in_channels, num_classes,
                 stages,
                 init_channels,
                 stage_channels,
                 feature_channels,
                 classifier_channels,
                 dropout_p):

        features = OrderedDict()
        features["stem"] = ConvBnAct2d(in_channels, init_channels, 3,
                                       padding=1,
                                       stride=2,
                                       activation=nn.Hardswish)
        for idx, stage in enumerate(stages):
            features[f"stage{idx + 1}"] = nn.Sequential(*stage)
        features["tail"] = ConvBnAct2d(stage_channels, feature_channels, 1,
                                       activation=nn.Hardswish)
        features = nn.Sequential(features)

        classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_channels, classifier_channels),
            nn.Hardswish(),
            nn.Dropout(p=dropout_p),
            nn.Linear(classifier_channels, num_classes),
        )

        super().__init__(OrderedDict([
            ('features', features),
            ('classifier', classifier),
        ]))


class MobileNetV3Small(MobileNetV3):
    def __init__(self, in_channels, num_classes, width_multiplier: float = 1.):
        def c(channels): return round_channels(channels * width_multiplier)

        IR = InvertedResidual  # for redability
        stages = [
            [   # stage 1
                IR(c(16), c(16), 3, 2, 1, activation=nn.ReLU), ],
            [   # stage 2
                IR(c(16), c(24), 3, 2, 4.5, activation=nn.ReLU, use_se=False),
                IR(c(24), c(24), 3, 1, 3.67, activation=nn.ReLU, use_se=False), ],
            [   # stage 3
                IR(c(24), c(40), 5, 2, 4),
                IR(c(40), c(40), 5, 1, 6),
                IR(c(40), c(40), 5, 1, 6), ],
            [   # stage 4
                IR(c(40), c(48), 5, 1, 3),
                IR(c(48), c(48), 5, 1, 3), ],
            [   # stage 5
                IR(c(48), c(96), 5, 2, 6),
                IR(c(96), c(96), 5, 1, 6),
                IR(c(96), c(96), 5, 1, 6), ],
        ]
        super(MobileNetV3Small, self).__init__(
            in_channels, num_classes,
            stages=stages,
            init_channels=c(16),
            stage_channels=c(96),
            feature_channels=c(576),
            classifier_channels=max(c(1024), 1024),
            dropout_p=0.1,
        )


class MobileNetV3Large(MobileNetV3):
    def __init__(self, in_channels, num_classes, width_multiplier: float = 1.):
        def c(channels): return round_channels(channels * width_multiplier)

        IR = InvertedResidual  # for redability
        stages = [
            [   # stage 1
                IR(c(16), c(16), 3, 1, 1, activation=nn.ReLU, use_se=False), ],
            [   # stage 2
                IR(c(16), c(24), 3, 2, 4, activation=nn.ReLU, use_se=False),
                IR(c(24), c(24), 3, 1, 3, activation=nn.ReLU, use_se=False), ],
            [   # stage 3
                IR(c(24), c(40), 5, 2, 3, activation=nn.ReLU),
                IR(c(40), c(40), 5, 1, 3, activation=nn.ReLU),
                IR(c(40), c(40), 5, 1, 3, activation=nn.ReLU), ],
            [   # stage 4
                IR(c(40), c(80), 3, 2, 6, use_se=False),
                IR(c(80), c(80), 3, 1, 2.5, use_se=False),
                IR(c(80), c(80), 3, 1, 2.3, use_se=False),
                IR(c(80), c(80), 3, 1, 2.3, use_se=False), ],
            [   # stage 5
                IR(c(80), c(112), 3, 1, 6),
                IR(c(112), c(112), 3, 1, 6), ],
            [   # stage 6
                IR(c(112), c(160), 5, 2, 6),
                IR(c(160), c(160), 5, 1, 6),
                IR(c(160), c(160), 5, 1, 6), ],
        ]
        super(MobileNetV3Large, self).__init__(
            in_channels, num_classes,
            stages=stages,
            init_channels=c(16),
            stage_channels=c(160),
            feature_channels=c(960),
            classifier_channels=max(c(1280), 1280),
            dropout_p=0.2,
        )


class InvertedResidual(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int,
                 stride: int,
                 expansion: int,
                 activation: type = nn.Hardswish,
                 use_se: bool = True):
        super(InvertedResidual, self).__init__()

        self.activation_after_se = use_se and expansion != 1

        width = round_by(expansion * in_channels)

        self.conv1 = (
            ConvBnAct2d(in_channels, width, 1, activation=activation)
            if expansion != 1 else nn.Identity()
        )

        self.conv2 = ConvBn2d(width, width, kernel_size,
                              padding=kernel_size // 2,
                              stride=stride,
                              groups=width)
        self.act2 = activation()
        self.se = (
            SqueezeExcitation(width, width)
            if use_se else nn.Identity()
        )

        self.conv3 = ConvBn2d(width, out_channels, 1)

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        if not self.activation_after_se:
            x = self.act2(x)
        x = self.se(x)
        if self.activation_after_se:
            x = self.act2(x)
        x = self.conv3(x)
        if input.shape == x.shape:
            x = x + input
        return x


class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=4):
        super().__init__()
        mid_channels = round_by(in_channels / reduction)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, 1)

    def forward(self, input):
        x = F.adaptive_avg_pool2d(input, 1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return input * F.hardsigmoid(x, inplace=True)
