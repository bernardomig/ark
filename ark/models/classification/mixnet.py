from typing import List, Tuple, Union
from collections import OrderedDict
import torch
from torch import nn
from torch.nn import functional as F

from ark.nn import Swish
from ark.nn.easy import ConvBn2d, ConvBnReLU2d, ConvBnSwish2d
from ark.nn.utils import round_channels


def mixnet_s(in_channels, num_classes):
    r"""MixNet S

    See :class:`~ark.models.classification.mixnet.MixNet` for details.
    """
    return MixNetS(in_channels, num_classes)


def mixnet_m(in_channels, num_classes):
    r"""MixNet M

    See :class:`~ark.models.classification.mixnet.MixNet` for details.
    """
    return MixNetM(in_channels, num_classes)


def mixnet_l(in_channels, num_classes):
    r"""MixNet L

    See :class:`~ark.models.classification.mixnet.MixNet` for details.
    """
    return MixNetM(in_channels, num_classes, width_multiplier=1.3)


class MixNet(nn.Sequential):
    r"""MixNets implementation from the
    `"MixConv: Mixed Depthwise Convolutional Kernels": 
    <https://arxiv.org/abs/1907.09595>` paper.

    Args:
        in_channels (int): the input channels
        num_classes (int): the number of the output classification classes
        stages: (list of list of :obj:`nn.Module`): the configuration for the 
            stages of the network, excluding the stem. 
        init_channels (int): the output channels of the stem conv
        stage_channels (int): the output channels of all the stages
        feature_channels (int): the desired output channels of the feature 
            encoder
    """

    def __init__(self, in_channels: int, num_classes: int,
                 stages: List[List[nn.Module]],
                 init_channels: int,
                 stage_channels: int,
                 feature_channels: int,
                 dropout_p: float = 0.1):
        features = OrderedDict()
        features["stem"] = ConvBnReLU2d(in_channels, init_channels, 3, padding=1, stride=2)
        for idx, stage in enumerate(stages):
            features[f"stage{idx+ 1}"] = nn.Sequential(*stage)
        features["tail"] = ConvBnReLU2d(stage_channels, feature_channels, 1)
        features = nn.Sequential(features)

        classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout(p=dropout_p),
            nn.Flatten(),
            nn.Linear(feature_channels, num_classes)
        )

        super().__init__(OrderedDict([
            ('features', features),
            ('classifier', classifier),
        ]))


class MixNetS(MixNet):
    r"""Small version of the MixNet.

    See :class:`~ark.models.classification.mixnet.MixNet` for details.
    """

    def __init__(self, in_channels, num_classes, width_multiplier: float = 1.):
        def c(channels): return round_channels(channels * width_multiplier)

        IR = InvertedResidual  # for redability
        stages = [
            [   # stage 1
                IR(c(16), c(16), 3, expansion=1, activation='relu', use_se=False), ],
            [   # stage 2
                IR(c(16), c(24), 3, stride=2, activation='relu', groups=(2, 2), use_se=False),
                IR(c(24), c(24), 3, activation='relu', groups=(2, 2), expansion=3, use_se=False), ],
            [   # stage 3
                IR(c(24), c(40), [3, 5, 7], stride=2, se_reduction=2),
                IR(c(40), c(40), [3, 5], groups=(2, 2), se_reduction=2),
                IR(c(40), c(40), [3, 5], groups=(2, 2), se_reduction=2),
                IR(c(40), c(40), [3, 5], groups=(2, 2), se_reduction=2), ],
            [   # stage 4
                IR(c(40), c(80), [3, 5, 7], stride=2, groups=(1, 2)),
                IR(c(80), c(80), [3, 5], groups=(1, 2)),
                IR(c(80), c(80), [3, 5], groups=(1, 2)), ],
            [   # stage 5
                IR(c(80), c(120), [3, 5, 7], groups=(2, 2), se_reduction=2),
                IR(c(120), c(120), [3, 5, 7, 9], groups=(2, 2), expansion=3, se_reduction=2),
                IR(c(120), c(120), [3, 5, 7, 9], groups=(2, 2), expansion=3, se_reduction=2), ],
            [   # stage 6
                IR(c(120), c(200), [3, 5, 7, 9, 11], stride=2, se_reduction=2),
                IR(c(200), c(200), [3, 5, 7, 9], groups=(1, 2), se_reduction=2),
                IR(c(200), c(200), [3, 5, 7, 9], groups=(1, 2), se_reduction=2), ],
        ]
        super(MixNetS, self).__init__(in_channels, num_classes,
                                      stages=stages,
                                      init_channels=c(16),
                                      stage_channels=c(200),
                                      feature_channels=1536)


class MixNetM(MixNet):
    r"""Medium version of the MixNet.

    See :class:`~ark.models.classification.mixnet.MixNet` for details.
    """

    def __init__(self, in_channels, num_classes, width_multiplier: float = 1.0):
        def c(channels): return round_channels(channels * width_multiplier)

        IR = InvertedResidual  # for redability
        stages = [
            [   # stage 1
                IR(c(24), c(24), 3, expansion=1, activation='relu', use_se=False), ],
            [   # stage 2
                IR(c(24), c(32), [3, 5, 7], stride=2, groups=(2, 2), activation='relu', use_se=False),
                IR(c(32), c(32), 3, groups=(2, 2), expansion=3, activation='relu', use_se=False), ],
            [   # stage 3
                IR(c(32), c(40), [3, 5, 7, 9], stride=2, se_reduction=2),
                IR(c(40), c(40), [3, 5], groups=(2, 2), se_reduction=2),
                IR(c(40), c(40), [3, 5], groups=(2, 2), se_reduction=2),
                IR(c(40), c(40), [3, 5], groups=(2, 2), se_reduction=2), ],
            [   # stage 4
                IR(c(40), c(80), [3, 5, 7], stride=2),
                IR(c(80), c(80), [3, 5, 7, 9], groups=(2, 2)),
                IR(c(80), c(80), [3, 5, 7, 9], groups=(2, 2)),
                IR(c(80), c(80), [3, 5, 7, 9], groups=(2, 2)), ],
            [   # stage 5
                IR(c(80), c(120), 3, se_reduction=2),
                IR(c(120), c(120), [3, 5, 7, 9], groups=(2, 2), expansion=3, se_reduction=2),
                IR(c(120), c(120), [3, 5, 7, 9], groups=(2, 2), expansion=3, se_reduction=2),
                IR(c(120), c(120), [3, 5, 7, 9], groups=(2, 2), expansion=3, se_reduction=2), ],
            [   # stage 6
                IR(c(120), c(200), [3, 5, 7, 9], stride=2, se_reduction=2),
                IR(c(200), c(200), [3, 5, 7, 9], groups=(1, 2), se_reduction=2),
                IR(c(200), c(200), [3, 5, 7, 9], groups=(1, 2), se_reduction=2),
                IR(c(200), c(200), [3, 5, 7, 9], groups=(1, 2), se_reduction=2), ],
        ]
        super(MixNetM, self).__init__(in_channels, num_classes,
                                      stages=stages,
                                      init_channels=c(24),
                                      stage_channels=c(200),
                                      feature_channels=1536)


class InvertedResidual(nn.Module):
    r"""Inverted residual block for MixNets"""

    def __init__(self, in_channels: int, out_channels: int, kernel_sizes: Union[int, List[int]],
                 stride: int = 1,
                 expansion: int = 6,
                 groups: Tuple[int, int] = (1, 1),
                 use_se: bool = True,
                 se_reduction: int = 4,
                 activation: str = 'swish'):
        super(InvertedResidual, self).__init__()

        assert not (activation == 'relu' and use_se), \
            "can only use the SE block if activation is swish"

        width = in_channels * expansion

        self.conv1 = (
            (ConvBnSwish2d if activation == 'swish' else ConvBnReLU2d)(in_channels, width, 1, groups=groups[0])
            if expansion != 1 else nn.Identity()
        )

        self.conv2 = (MixConvBnSwish2d if activation == 'swish' else MixConvBnReLU2d)(
            width, width, kernel_sizes, stride=stride)
        self.se = SqueezeExcitation(width, width, int(in_channels / se_reduction)) if use_se else nn.Identity()
        self.conv3 = ConvBn2d(width, out_channels, 1, groups=groups[1])

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.se(x)
        x = self.conv3(x)
        if x.shape == input.shape:
            x = x + input
        return x


class MixConvBnReLU2d(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_sizes: Union[int, List[int]],
                 stride: int = 1,):
        super(MixConvBnReLU2d, self).__init__(OrderedDict([
            ('conv', (MixConv2d(in_channels, out_channels, kernel_sizes, stride=stride, bias=False)
                      if isinstance(kernel_sizes, list)
                      else nn.Conv2d(in_channels, out_channels, kernel_sizes, padding=kernel_sizes//2, stride=stride, groups=in_channels, bias=False))),
            ('bn', nn.BatchNorm2d(out_channels)),
            ('relu', nn.ReLU(inplace=True)),
        ]))


class MixConvBnSwish2d(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_sizes: List[int],
                 stride: int = 1,):
        super(MixConvBnSwish2d, self).__init__(OrderedDict([
            ('conv', (MixConv2d(in_channels, out_channels, kernel_sizes, stride=stride, bias=False)
                      if isinstance(kernel_sizes, list)
                      else nn.Conv2d(in_channels, out_channels, kernel_sizes, padding=kernel_sizes//2, stride=stride, groups=in_channels, bias=False))),
            ('bn', nn.BatchNorm2d(out_channels)),
            ('swish', Swish(inplace=True)),
        ]))


class MixConv2d(nn.ModuleList):
    def __init__(self, in_channels: int, out_channels: int, kernel_sizes: List[int], stride: int = 1, bias: int = True):
        super().__init__([
            nn.Conv2d(in_channels // len(kernel_sizes), out_channels // len(kernel_sizes), kernel_size,
                      padding=kernel_size // 2,
                      stride=stride,
                      groups=out_channels // len(kernel_sizes),
                      bias=bias
                      )
            for kernel_size in kernel_sizes
        ])

    def forward(self, input):
        xs = torch.chunk(input, len(self), dim=1)
        return torch.cat([conv(x) for conv, x in zip(self.children(), xs)], dim=1)


class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(SqueezeExcitation, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.act = Swish(inplace=True)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, 1)

    def forward(self, input):
        x = F.adaptive_avg_pool2d(input, 1)
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        return input * torch.sigmoid_(x)
