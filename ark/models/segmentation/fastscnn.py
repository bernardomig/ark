from collections import OrderedDict
import torch
from torch import nn
from torch.nn import functional as F

from ark.nn.easy import ConvBn2d, ConvBnReLU2d
from ark.utils.hub import register_model


@register_model(
    cityscapes={'in_channels': 3, 'out_channels': 19,
                'state_dict': 'https://files.deeplar.tk/ark-weights/fastscnn-cityscapes-d94d460efe.pt'},
)
def fastscnn(in_channels, out_channels):
    return FastSCNN(in_channels, out_channels)


class FastSCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # The Learning to Downsample module
        # It encodes low level features in a efficient way
        # It is composed by three convolutional layers, where the first one is
        # a regular conv and the other two are depthwise separable conv
        # layers.
        # The first convolutional layer is a regular conv because there is no
        # advantage in using a ds conv in such small number of channels.
        # All layers are a spatial kernel of 3x3 and have a stride of 2 for a total downsample of 8 times.
        # Also, there is no nonlinearity between the depthwise and pointwise conv.
        self.downsample = nn.Sequential(
            ConvBnReLU2d(in_channels, 32, 3, padding=1, stride=1),
            ConvBn2d(32, 32, 3, padding=1, stride=2, groups=32),
            ConvBnReLU2d(32, 48, 1),
            ConvBn2d(48, 48, 3, padding=1, stride=2, groups=48),
            ConvBnReLU2d(48, 64, 1),
        )

        # The Global Feature Extractor module is aimed at capturing the global
        # context for the task of image segmentation.
        # This module directly takes the 1/8 downsampled output of the
        # Learning to Downsample module, performs a feature encoding using the
        # MobileNet bottleneck residual block and then performs a pyramid pooling
        # at the end to aggregate the different region-based context information.
        self.features = nn.Sequential(
            BottleneckModule(64, 64, expansion=6, repeats=3, stride=2),
            BottleneckModule(64, 96, expansion=6, repeats=3, stride=2),
            BottleneckModule(96, 128, expansion=6, repeats=3, stride=1),
            PyramidPoolingModule(128, 128)
        )

        # The Feature Fusion adds the low-resolution features from the
        # Global Feature Encoder and the high-resolution features from the
        # Learning to Downsample Module.
        self.fusion = FeatureFusionModule((128, 64), 128, scale_factor=4)

        # The classifier discriminates the classes from the features produced
        # by fusion module.
        self.classifier = Classifier(128, out_channels)

    def forward(self, input):
        downsample = self.downsample(input)
        features = self.features(downsample)
        fusion = self.fusion(features, downsample)
        classes = self.classifier(fusion)

        return F.interpolate(classes, size=input.shape[2:],
                             mode='bilinear', align_corners=True)


def Classifier(in_channels, out_channels):
    return nn.Sequential(
        ConvBn2d(in_channels, in_channels, 3, padding=1, groups=in_channels),
        ConvBnReLU2d(in_channels, in_channels, 1),
        ConvBn2d(in_channels, in_channels, 3, padding=1, groups=in_channels),
        ConvBnReLU2d(in_channels, in_channels, 1),
        nn.Dropout(0.1),
        nn.Conv2d(in_channels, out_channels, kernel_size=1),
    )


class FeatureFusionModule(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor):
        super().__init__()
        lowres_channels, highres_channels = in_channels

        self.lowres = nn.Sequential(
            ConvBnReLU2d(lowres_channels, lowres_channels, 3,
                         padding=scale_factor, dilation=scale_factor,
                         groups=lowres_channels),
            ConvBn2d(lowres_channels, out_channels, 1),
        )

        self.highres = ConvBn2d(highres_channels, out_channels, 1)

    def forward(self, lowres, highres):
        lowres = F.interpolate(lowres, size=highres.shape[2:], mode='bilinear', align_corners=True)
        lowres = self.lowres(lowres)
        highres = self.highres(highres)
        return F.relu(lowres + highres)


class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels, out_channels, pyramids=(1, 2, 3, 6)):
        super().__init__()

        self.pyramids = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                ConvBnReLU2d(in_channels, in_channels // len(pyramids), 1),
            )
            for bin in pyramids
        ])

        self.conv = ConvBnReLU2d(in_channels * 2, out_channels, 1)

    def forward(self, input):
        pools = [
            F.interpolate(pool(input), size=input.shape[2:],
                          mode='bilinear', align_corners=True)
            for pool in self.pyramids.children()]
        x = torch.cat([input, *pools], dim=1)
        return self.conv(x)


def BottleneckModule(in_channels, out_channels, expansion, repeats=1, stride=1):
    layers = [
        BottleneckBlock(in_channels, out_channels,
                        expansion=expansion, stride=stride)
    ]
    for _ in range(1, repeats):
        layers.append(
            BottleneckBlock(out_channels, out_channels, expansion=expansion)
        )
    return nn.Sequential(*layers)


class BottleneckBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, expansion=6):
        super().__init__()

        expansion_channels = expansion * in_channels
        self.conv1 = ConvBnReLU2d(in_channels, expansion_channels, 1)
        self.conv2 = ConvBnReLU2d(expansion_channels, expansion_channels, 3,
                                  padding=1, stride=stride,
                                  groups=expansion_channels)
        self.conv3 = ConvBn2d(expansion_channels, out_channels, 1)

    def forward(self, input: torch.Tensor):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)

        if x.shape == input.shape:
            x = x + input

        return F.relu(x)
