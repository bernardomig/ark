from collections import OrderedDict
import torch
from torch import nn


class ConvBnReLU2d(nn.Sequential):
    r"""The standard conv+bn+relu started in the VGG models
    and used in almost all modern network architectures.

    As usual, the convolution operation includes the bias term and
    the relu operation is performed inplace.

    The arguments are the same as in the convolution operation.
    See :class:`torch.nn.Conv2d`.
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 padding=0,
                 stride=1,
                 dilation=1,
                 groups=1):
        super(ConvBnReLU2d, self).__init__(OrderedDict([
            ('conv', nn.Conv2d(in_channels, out_channels, kernel_size,
                               padding=padding,
                               stride=stride,
                               dilation=dilation,
                               groups=groups,
                               bias=False)),
            ('bn', nn.BatchNorm2d(out_channels)),
            ('relu', nn.ReLU(inplace=True)),
        ]))


class ConvReLU2d(nn.Sequential):
    r"""The standard conv+relu started in the VGG models
    and used in almost all modern network architectures.

    As usual, the convolution operation includes the bias term and
    the relu operation is performed inplace.

    The arguments are the same as in the convolution operation.
    See :class:`torch.nn.Conv2d`.
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 padding=0,
                 stride=1,
                 dilation=1,
                 groups=1):
        super(ConvReLU2d, self).__init__(OrderedDict([
            ('conv', nn.Conv2d(in_channels, out_channels, kernel_size,
                               padding=padding,
                               stride=stride,
                               dilation=dilation,
                               groups=groups)),
            ('relu', nn.ReLU(inplace=True)),
        ]))


class ConvBn2d(nn.Sequential):
    r"""The standard conv+bn.

    As usual, the convolution operation does not include the bias term.

    The arguments are the same as in the convolution operation.
    See :class:`torch.nn.Conv2d`.
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 padding=0,
                 stride=1,
                 dilation=1,
                 groups=1):
        super(ConvBn2d, self).__init__(OrderedDict([
            ('conv', nn.Conv2d(in_channels, out_channels, kernel_size,
                               padding=padding,
                               stride=stride,
                               dilation=dilation,
                               groups=groups,
                               bias=False)),
            ('bn', nn.BatchNorm2d(out_channels)),
        ]))
