from collections import OrderedDict
import torch
from torch import nn
from torch.nn import functional as F

from ark.nn.easy import ConvBnReLU2d
from ark.utils.hub import register_model


@register_model(
    cityscapes={'in_channels': 3, 'out_channels': 19,
                'state_dict': 'pspnet_resnet18-cityscapes-74cdcf03c8.pt'},
)
def pspnet_resnet18(in_channels, out_channels):
    from ark.models.classification.resnet import resnet18
    model = resnet18(in_channels, 1)
    model.replace_stride_with_dilation(8)
    features = model.features
    return PSPNet(features, out_channels, feature_channels=512, ppm_channels=128)


@register_model(
    cityscapes={'in_channels': 3, 'out_channels': 19,
                'state_dict': 'pspnet_resnet34-cityscapes-6b341e14a4.pt'},
)
def pspnet_resnet34(in_channels, out_channels):
    from ark.models.classification.resnet import resnet34
    model = resnet34(in_channels, 1)
    model.replace_stride_with_dilation(8)
    features = model.features
    return PSPNet(features, out_channels, feature_channels=512, ppm_channels=128)


def pspnet_resnet50(in_channels, out_channels):
    from ark.models.classification.resnet import resnet50
    model = resnet50(in_channels, 1)
    model.replace_stride_with_dilation(8)
    features = model.features
    return PSPNet(features, out_channels, feature_channels=2048, ppm_channels=256)


class PSPNet(nn.Module):
    def __init__(self, feature_extractor, out_channels, feature_channels, ppm_channels):
        super().__init__()

        self.features = feature_extractor
        self.ppm = PyramidPooling(feature_channels, ppm_channels, bins=[1, 2, 3, 6])
        self.classifier = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(ppm_channels, out_channels, 1),
        )

    def forward(self, input):
        x = self.features(input)
        x = self.ppm(x)
        x = self.classifier(x)
        return F.interpolate(x, size=input.shape[2:], mode='bilinear', align_corners=True)


class PyramidPooling(nn.Module):
    def __init__(self, in_channels, out_channels, bins=[1, 2, 3, 6]):
        super(PyramidPooling, self).__init__()
        self.pyramids = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=bin),
                ConvBnReLU2d(in_channels, in_channels, 1),
            )
            for bin in bins
        ])
        self.bottleneck = ConvBnReLU2d(in_channels * (len(bins) + 1), out_channels, 1)

    def forward(self, input):
        pyramids = torch.cat([
            F.interpolate(pyramid(input), size=input.shape[2:], mode='bilinear', align_corners=True)
            for pyramid in self.pyramids
        ], dim=1)

        x = torch.cat([input, pyramids], dim=1)
        return self.bottleneck(x)
