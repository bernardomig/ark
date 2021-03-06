from collections import OrderedDict
from torch import nn

from ark.nn.easy import ConvBn2d
from ark.nn.utils import round_channels
from ark.utils.hub import register_model

__all__ = [
    'MobileNetV2',
    'mobilenetv2_1_0', 'mobilenetv2_0_75', 'mobilenetv2_0_50',
    'mobilenetv2_0_35', 'mobilenetv2_0_25', 'mobilenetv2_0_10',
]


@register_model(
    imagenet1k={'in_channels': 3, 'num_classes': 1000,
                'state_dict': 'mobilenetv2_1_0-imagenet-bf3dd96df3.pt'},
)
def mobilenetv2_1_0(in_channels, num_classes):
    r"""MobileNetV2 with width 1.0

    See :class:`~ark.models.classification.mobilenetv2.MobileNetV2` for details.
    """
    return MobileNetV2(in_channels, num_classes, width_multiplier=1)


@register_model(
    imagenet1k={'in_channels': 3, 'num_classes': 1000,
                'state_dict': 'mobilenetv2_0_75-imagenet-34c9a7dc0c.pt'},
)
def mobilenetv2_0_75(in_channels, num_classes):
    r"""MobileNetV2 with width 0.75

    See :class:`~ark.models.classification.mobilenetv2.MobileNetV2` for details.
    """
    return MobileNetV2(in_channels, num_classes, width_multiplier=0.75)


@register_model(
    imagenet1k={'in_channels': 3, 'num_classes': 1000,
                'state_dict': 'mobilenetv2_0_50-imagenet-228a3ef7f5.pt'},
)
def mobilenetv2_0_50(in_channels, num_classes):
    r"""MobileNetV2 with width 0.50

    See :class:`~ark.models.classification.mobilenetv2.MobileNetV2` for details.
    """
    return MobileNetV2(in_channels, num_classes, width_multiplier=0.5)


@register_model(
    imagenet1k={'in_channels': 3, 'num_classes': 1000,
                'state_dict': 'mobilenetv2_0_35-imagenet-2aafa73305.pt'},
)
def mobilenetv2_0_35(in_channels, num_classes):
    r"""MobileNetV2 with width 0.35

    See :class:`~ark.models.classification.mobilenetv2.MobileNetV2` for details.
    """
    return MobileNetV2(in_channels, num_classes, width_multiplier=0.35)


@register_model(
    imagenet1k={'in_channels': 3, 'num_classes': 1000,
                'state_dict': 'mobilenetv2_0_25-imagenet-1657878274.pt'},
)
def mobilenetv2_0_25(in_channels, num_classes):
    r"""MobileNetV2 with width 0.25

    See :class:`~ark.models.classification.mobilenetv2.MobileNetV2` for details.
    """
    return MobileNetV2(in_channels, num_classes, width_multiplier=0.25)


@register_model(
    imagenet1k={'in_channels': 3, 'num_classes': 1000,
                'state_dict': 'mobilenetv2_0_10-imagenet-00c0863e22.pt'},
)
def mobilenetv2_0_10(in_channels, num_classes):
    r"""MobileNetV2 with width 0.10

    See :class:`~ark.models.classification.mobilenetv2.MobileNetV2` for details.
    """
    return MobileNetV2(in_channels, num_classes, width_multiplier=0.10)


class MobileNetV2(nn.Sequential):
    r"""MobilenetV2 implementation from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks":
    <https://arxiv.org/abs/1801.04381>`_ paper.

    Args:
        in_channels (int): the input channels
        num_classes (int): the number of the output classification classes
        width_multiplier (float): the width multiplier hyperparameter. Default: 1
    """

    def __init__(self, in_channels, num_classes, width_multiplier=1.0):
        def c(channels):
            "channel number mapper"
            return round_channels(width_multiplier * channels,
                                  8 if width_multiplier > 0.1 else 4)

        def make_layer(in_channels, out_channels, num_blocks=1, expansion=6, stride=1):
            layers = [InvertedResidual(in_channels, out_channels,
                                       stride=stride, expansion=expansion)]
            for _ in range(1, num_blocks):
                layers += [InvertedResidual(out_channels, out_channels,
                                            stride=1, expansion=expansion)]
            return nn.Sequential(*layers)

        # the maximum number of channels in the features is 1280
        out_channels = max(1280, c(1280))

        features = nn.Sequential(OrderedDict([
            ('head', ConvBnReLU62d(in_channels, c(32), 3, padding=1, stride=2)),
            ('layer1', make_layer(c(32), c(16), expansion=1)),
            ('layer2', make_layer(c(16), c(24), num_blocks=2, stride=2)),
            ('layer3', make_layer(c(24), c(32), num_blocks=3, stride=2)),
            ('layer4', make_layer(c(32), c(64), num_blocks=4, stride=2)),
            ('layer5', make_layer(c(64), c(96), num_blocks=3, stride=1)),
            ('layer6', make_layer(c(96), c(160), num_blocks=3, stride=2)),
            ('layer7', make_layer(c(160), c(320), num_blocks=1, stride=1)),
            ('tail', ConvBnReLU62d(c(320), out_channels, 1)),
        ]))

        classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(),
            nn.Linear(out_channels, num_classes),
        )

        super(MobileNetV2, self).__init__(OrderedDict([
            ('features', features),
            ('classifier', classifier),
        ]))

    def replace_stride_with_dilation(self, output_stride):
        from itertools import chain
        assert output_stride in {8, 16, 32}
        features = self.features

        strides = [1 if output_stride == 8 else 2, 2 if output_stride == 32 else 1]
        dilation = [2 if output_stride == 8 else 1, 4 if output_stride == 8 else 2 if output_stride == 16 else 1]

        for i, block in enumerate(chain(features.layer4.children(), features.layer5.children())):
            block.replace_stride_with_dilation(stride=strides[0] if i == 0 else 1, dilation=dilation[0])
        for i, block in enumerate(chain(features.layer6.children(), features.layer7.children())):
            block.replace_stride_with_dilation(stride=strides[1] if i == 0 else 1, dilation=dilation[1])


class InvertedResidual(nn.Module):
    r"""Inverted Residual block of MobilenetV2.

    Args:
        in_channels (int): the input channels
        out_channels (int): the output channels
        stride (int): the stride of the block
        expansion (int): the expansion rate of the mid channels
    """

    def __init__(self, in_channels: int, out_channels: int,
                 stride: int = 1,
                 expansion: int = 6):
        super(InvertedResidual, self).__init__()

        hidden_channels = in_channels * expansion
        self.conv1 = (
            ConvBnReLU62d(in_channels, hidden_channels, 1)
            if expansion != 1
            else nn.Identity())
        self.conv2 = ConvBnReLU62d(hidden_channels, hidden_channels, 3,
                                   padding=1, stride=stride,
                                   groups=hidden_channels)
        self.conv3 = ConvBn2d(hidden_channels, out_channels, 1)

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        if input.shape == x.shape:
            x = x + input
        return x

    def replace_stride_with_dilation(self, stride, dilation):
        from torch.nn.modules.utils import _pair
        self.conv2.conv.stride = _pair(stride)
        self.conv2.conv.dilation = _pair(dilation)
        self.conv2.conv.padding = _pair(dilation)


class ConvBnReLU62d(nn.Sequential):
    r"""The conv+bn+relu6 of present on the mobilenetv2 architecture.

    As usual, the convolution operation includes the bias term and
    the relu operation is performed inplace.

    The arguments are the same as in the convolution operation.
    See :class:`torch.nn.Conv2d`.
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 padding=0, stride=1, groups=1):
        super().__init__(OrderedDict([
            ('conv', nn.Conv2d(in_channels, out_channels, kernel_size,
                               padding=padding, stride=stride,
                               groups=groups,
                               bias=False)),
            ('bn', nn.BatchNorm2d(out_channels)),
            ('relu', nn.ReLU6(inplace=True)),
        ]))
