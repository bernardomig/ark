from typing import List, Optional
from collections import OrderedDict
from functools import partial
from torch import nn

from ark.nn.easy import ConvBnReLU2d, ConvBn2d


def drn_a_18(in_channels=3, num_classes=1000):
    r"""DRN Variant A with 18 layers.

    See :class:`~ark.models.classification.drn.DRN` for details."""
    return DRN(in_channels, num_classes, 'A', BasicBlock,
               block_depth=[2, 2, 2, 2],
               init_channels=64,
               block_channels=[64, 128, 256, 512])


def drn_a_34(in_channels=3, num_classes=1000):
    r"""DRN Variant A with 34 layers.

    See :class:`~ark.models.classification.drn.DRN` for details."""
    return DRN(in_channels, num_classes, 'A', BasicBlock,
               block_depth=[3, 4, 6, 3],
               init_channels=[64, 128, 256, 512])


def drn_a_50(in_channels=3, num_classes=1000):
    r"""DRN Variant A with 50 layers.

    See :class:`~ark.models.classification.drn.DRN` for details."""
    return DRN(in_channels, num_classes, 'A', Bottleneck,
               block_depth=[3, 4, 6, 3],
               init_channels=64,
               block_channels=[256, 512, 1024, 2048])


def drn_c_26(in_channels=3, num_classes=1000):
    r"""DRN Variant C with 26 layers.

    See :class:`~ark.models.classification.drn.DRN` for details."""
    return DRN(in_channels, num_classes, 'C', BasicBlock,
               block_depth=[2, 2, 2, 2, 1, 1],
               init_channels=32,
               block_channels=[64, 128, 256, 512],
               last_channels=512)


def drn_c_42(in_channels=3, num_classes=1000):
    r"""DRN Variant C with 42 layers.

    See :class:`~ark.models.classification.drn.DRN` for details."""
    return DRN(in_channels, num_classes, 'C', BasicBlock,
               block_depth=[3, 4, 6, 3, 1, 1],
               init_channels=32,
               block_channels=[64, 128, 256, 512],
               last_channels=512)


def drn_c_58(in_channels=3, num_classes=1000):
    r"""DRN Variant C with 58 layers.

    See :class:`~ark.models.classification.drn.DRN` for details."""
    return DRN(in_channels, num_classes, 'C', Bottleneck,
               block_depth=[3, 4, 6, 3, 1, 1],
               init_channels=32,
               block_channels=[256, 512, 1024, 2048],
               last_channels=512)


def drn_d_22(in_channels=3, num_classes=1000):
    r"""DRN Variant D with 22 layers.

    See :class:`~ark.models.classification.drn.DRN` for details."""
    return DRN(in_channels, num_classes, 'D', BasicBlock,
               block_depth=[2, 2, 2, 2, 1, 1],
               init_channels=32,
               block_channels=[64, 128, 256, 512],
               last_channels=512)


def drn_d_24(in_channels=3, num_classes=1000):
    r"""DRN Variant D with 24 layers.

    See :class:`~ark.models.classification.drn.DRN` for details."""
    return DRN(in_channels, num_classes, 'D', BasicBlock,
               block_depth=[2, 2, 2, 2, 2, 2],
               init_channels=32,
               block_channels=[64, 128, 256, 512],
               last_channels=512)


def drn_d_38(in_channels=3, num_classes=1000):
    r"""DRN Variant D with 38 layers.

    See :class:`~ark.models.classification.drn.DRN` for details."""
    return DRN(in_channels, num_classes, 'D', BasicBlock,
               block_depth=[3, 4, 6, 3, 1, 1],
               init_channels=32,
               block_channels=[64, 128, 256, 512],
               last_channels=512)


def drn_d_40(in_channels=3, num_classes=1000):
    r"""DRN Variant D with 40 layers.

    See :class:`~ark.models.classification.drn.DRN` for details."""
    return DRN(in_channels, num_classes, 'D', BasicBlock,
               block_depth=[3, 4, 6, 3, 2, 2],
               init_channels=32,
               block_channels=[64, 128, 256, 512],
               last_channels=512)


def drn_d_54(in_channels=3, num_classes=1000):
    r"""DRN Variant D with 54 layers.

    See :class:`~ark.models.classification.drn.DRN` for details."""
    return DRN(in_channels, num_classes, 'D', Bottleneck,
               block_depth=[3, 4, 6, 3, 1, 1],
               init_channels=32,
               block_channels=[256, 512, 1024, 2048],
               last_channels=512)


def drn_d_56(in_channels=3, num_classes=1000):
    r"""DRN Variant D with 56 layers.

    See :class:`~ark.models.classification.drn.DRN` for details."""
    return DRN(in_channels, num_classes, 'D', Bottleneck,
               block_depth=[3, 4, 6, 3, 2, 2],
               init_channels=32,
               block_channels=[256, 512, 1024, 2048],
               last_channels=512)


def drn_d_105(in_channels=3, num_classes=1000):
    r"""DRN Variant D with 105 layers.

    See :class:`~ark.models.classification.drn.DRN` for details."""
    return DRN(in_channels, num_classes, 'D', Bottleneck,
               block_depth=[3, 4, 23, 3, 1, 1],
               init_channels=32,
               block_channels=[256, 512, 1024, 2048],
               last_channels=512)


def drn_d_107(in_channels=3, num_classes=1000):
    r"""DRN Variant D with 107 layers.

    See :class:`~ark.models.classification.drn.DRN` for details."""
    return DRN(in_channels, num_classes, 'D', Bottleneck,
               block_depth=[3, 4, 23, 3, 2, 2],
               init_channels=32,
               block_channels=[256, 512, 1024, 2048],
               last_channels=512)


class DRN(nn.Sequential):
    r"""Dilated Residual Networks implementation from 
    `"Dilated Residual Networks": <https://arxiv.org/abs/1705.09914>`_ paper.

    This architecture is an improved resnet designed to solve some of the 
    problems that come from replacing the stride with the dilation. The authors
    propose changing the stem of the resnet and adding two layers after the
    original resnet layers (layer 5 and 6), as well modifying the stem to reduce
    the early downsampling (from stride 4 to 2), and adding a stride 2 to the 
    first layer.

    As an ablation study, this work proposes 4 variants of the DRN:

    - Variant 'A': the vanilla dilated resnet;
    - Variant 'B': add two layers of residual blocks to the end of the resnet, 
        and modify the stem, substituting the max-pooling with
        two basic residual blocks;
    - Variant 'C': same as the 'B' variant, but do not use the residual 
        connection on the added layers;
    - Variant 'D': add two layers of conv-bn-relu layers to the end of the resnet,
        and modify the stem, substituting the max-pooling with two 
        conv-bn-relu blocks.

    Args:
        in_channels (int): the input channels
        num_classes (int): the number of the output classification classes
        variant (str): the identifier of the model variant. Can be 
            either 'A', 'B', 'C' or 'D'
        block (:obj:~nn.Module): the block type used in the resnet layers
        block_depth (list of int): the number of blocks at each layer.
        init_channels (int): the number of output features of the stem block
        block_channels (list of int): the width of each layer.
        last_channels (int): number of channels in the last two layers. 
            Ignored in the case of the 'A' variant
    """

    def __init__(self, in_channels: int, num_classes: int, variant: str,
                 block: nn.Module,
                 block_depth: List[int],
                 init_channels: int,
                 block_channels: List[int],
                 last_channels: Optional[int] = None):

        assert variant in {'A', 'B', 'C', 'D'}, \
            "architecture can only be one of A,B,C,D"

        def make_layer(in_channels, out_channels, num_blocks, block, stride=1):
            layers = [block(in_channels, out_channels, stride=stride)]
            for _ in range(1, num_blocks):
                layers += [block(out_channels, out_channels)]
            return nn.Sequential(*layers)

        features = OrderedDict()

        if variant == 'A':
            features['stem'] = nn.Sequential(
                ConvBnReLU2d(in_channels, init_channels, 7, padding=3, stride=1),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
        elif variant == 'B' or variant == 'C':
            features['stem'] = nn.Sequential(
                ConvBnReLU2d(in_channels, init_channels//2, 7, padding=3, stride=1),
                BasicBlock(init_channels//2, init_channels//2),
                BasicBlock(init_channels//2, init_channels, stride=2),
            )
        else:  # arch == 'D'
            features['stem'] = nn.Sequential(
                ConvBnReLU2d(in_channels, init_channels//2, 7, padding=3, stride=1),
                ConvBnReLU2d(init_channels//2, init_channels//2, 3, padding=1),
                ConvBnReLU2d(init_channels//2, init_channels, 3, padding=1, stride=2),
            )

        features['layer1'] = make_layer(init_channels, block_channels[0],
                                        block_depth[0], block, stride=2)
        features['layer2'] = make_layer(block_channels[0], block_channels[1],
                                        block_depth[1], block, stride=2)
        features['layer3'] = make_layer(block_channels[1], block_channels[2],
                                        block_depth[2], partial(block, dilation=2))
        features['layer4'] = make_layer(block_channels[2], block_channels[3],
                                        block_depth[3], partial(block, dilation=4))

        if variant == 'B':
            features['layer5'] = make_layer(block_channels[3], last_channels, num_blocks=block_depth[4],
                                            block=partial(block, dilation=2, use_residual=True))
            features['layer6'] = make_layer(last_channels, last_channels, num_blocks=block_depth[5],
                                            block=partial(block, dilation=1, use_residual=True))
        elif variant == 'C':
            features['layer5'] = make_layer(block_channels[3], last_channels, num_blocks=block_depth[4],
                                            block=partial(BasicBlock, dilation=2, use_residual=False))
            features['layer6'] = make_layer(last_channels, last_channels, num_blocks=block_depth[5],
                                            block=partial(BasicBlock, dilation=1, use_residual=False))
        elif variant == 'D':
            features['layer5'] = make_layer(block_channels[3], last_channels, num_blocks=block_depth[4],
                                            block=partial(ConvBnReLU2d, kernel_size=3, padding=2, dilation=2))
            features['layer6'] = make_layer(last_channels, last_channels, num_blocks=block_depth[5],
                                            block=partial(ConvBnReLU2d, kernel_size=3, padding=1, dilation=1))

        classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(last_channels, num_classes),
        )

        super(DRN, self).__init__(OrderedDict([
            ('features', nn.Sequential(features)),
            ('classifier', classifier),
        ]))


class BasicBlock(nn.Module):
    r"""Basic residual block of the resnet model, which optionally can discard 
    the residual connection.
    """

    use_residual: bool

    def __init__(self, in_channels, out_channels,
                 stride=1, dilation=1, use_residual=True):
        super(BasicBlock, self).__init__()

        self.use_residual = use_residual

        self.conv1 = ConvBnReLU2d(in_channels, out_channels, 3,
                                  padding=dilation, stride=stride,
                                  dilation=dilation)
        self.conv2 = ConvBn2d(out_channels, out_channels, 3,
                              padding=dilation, dilation=dilation)

        if self.use_residual:
            self.downsample = (
                ConvBn2d(in_channels, out_channels, 1, stride=stride)
                if in_channels != out_channels or stride != 1
                else nn.Identity()
            )

        self.activation = nn.ReLU(inplace=True)

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        if self.use_residual:
            residual = self.downsample(input)
            x = x + residual
        return self.activation(x)


class Bottleneck(nn.Module):
    r"""Basic bottleneck block of the resnet model, which optionally can discard 
    the residual connection.
    """

    use_residual: bool

    def __init__(self, in_channels, out_channels,
                 stride=1, dilation=1,
                 expansion=4,
                 use_residual=True):
        super(Bottleneck, self).__init__()

        self.use_residual = use_residual

        width = out_channels // expansion

        self.conv1 = ConvBnReLU2d(in_channels, width, 1)
        self.conv2 = ConvBnReLU2d(width, width, 3,
                                  padding=dilation,
                                  stride=stride,
                                  dilation=dilation)
        self.conv3 = ConvBn2d(width, out_channels, 1)

        if self.use_residual:
            self.downsample = (
                ConvBn2d(in_channels, out_channels, 1, stride=stride)
                if in_channels != out_channels or stride != 1
                else nn.Identity()
            )

        self.activation = nn.ReLU(inplace=True)

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        if self.use_residual:
            residual = self.downsample(input)
            x = x + residual
        return self.activation(x)
