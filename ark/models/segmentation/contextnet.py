from torch import nn
from torch.nn import functional as F

from ark.nn.easy import ConvBn2d, ConvBnReLU2d
from ark.utils.hub import register_model

__all__ = [
    'ContextNet',
    'contextnet12',
    'contextnet14',
    'contextnet18',
]


@register_model(
    cityscapes={'in_channels': 3, 'out_channels': 19,
                'state_dict': 'contextnet12-cityscapes-35f9d5cfc2.pt'},
)
def contextnet12(in_channels=3, out_channels=19):
    return ContextNet(in_channels, out_channels,
                      scale_factor=2)


@register_model(
    cityscapes={'in_channels': 3, 'out_channels': 19,
                'state_dict': 'contextnet14-cityscapes-15dc9302ca.pt'},
    bdd100k={'in_channels': 3, 'out_channels': 19,
             'state_dict': 'contextnet14-bdd100k-dec859f5bf.pt'},
)
def contextnet14(in_channels, out_channels, width_multiplier=1):
    return ContextNet(in_channels, out_channels,
                      scale_factor=4,
                      width_multiplier=width_multiplier)


@register_model(
    cityscapes={'in_channels': 3, 'out_channels': 19,
                'state_dict': 'contextnet18-cityscapes-40bf30973d.pt'},
    bdd100k={'in_channels': 3, 'out_channels': 19,
             'state_dict': 'contextnet18-bdd100k-6dac20b713.pt'},
)
def contextnet18(in_channels=3, out_channels=19):
    return ContextNet(in_channels, out_channels,
                      scale_factor=8)


class ContextNet(nn.Module):

    def __init__(self, in_channels, out_channels,
                 scale_factor=4,
                 width_multiplier=1):
        super(ContextNet, self).__init__()

        self.scale_factor = scale_factor

        def c(channels): return int(channels * width_multiplier)

        self.spatial = nn.Sequential(
            ConvBnReLU2d(in_channels, c(32), 3, padding=1, stride=2),
            ConvBnReLU2d(c(32), c(32), kernel_size=3, padding=1, stride=2, groups=c(32)),
            ConvBnReLU2d(c(32), c(64), 1),
            ConvBnReLU2d(c(64), c(64), kernel_size=3, padding=1, stride=2, groups=c(64)),
            ConvBnReLU2d(c(64), c(128), 1),
            ConvBnReLU2d(c(128), c(128), kernel_size=3, padding=1, stride=1, groups=c(128)),
            ConvBnReLU2d(c(128), c(128), 1),
        )

        self.context = nn.Sequential(
            ConvBnReLU2d(in_channels, c(32), 3, padding=1, stride=2),
            BottleneckBlock(c(32), c(32), expansion=1),
            BottleneckBlock(c(32), c(32), expansion=6),
            LinearBottleneck(c(32), c(48), 3, stride=2),
            LinearBottleneck(c(48), c(64), 3, stride=2),
            LinearBottleneck(c(64), c(96), 2),
            LinearBottleneck(c(96), c(128), 2),
            ConvBnReLU2d(c(128), c(128), 3, padding=1),
        )

        self.feature_fusion = FeatureFusionModule((c(128), c(128)), c(128))

        self.classifier = Classifier(c(128), out_channels)

    def forward(self, input):
        spatial = self.spatial(input)

        context = F.interpolate(
            input, scale_factor=1 / self.scale_factor,
            mode='bilinear', align_corners=True)
        context = self.context(context)

        fusion = self.feature_fusion(context, spatial)

        classes = self.classifier(fusion)

        return F.interpolate(
            classes, size=input.shape[2:],
            mode='bilinear', align_corners=True)


def Classifier(in_channels, out_channels):
    return nn.Sequential(
        ConvBnReLU2d(in_channels, in_channels, 3, padding=1, groups=in_channels),
        ConvBnReLU2d(in_channels, in_channels, 1),
        ConvBnReLU2d(in_channels, in_channels, 3, padding=1, groups=in_channels),
        ConvBnReLU2d(in_channels, in_channels, 1),
        nn.Dropout(p=0.1),
        nn.Conv2d(in_channels, out_channels, 1),
    )


def LinearBottleneck(in_channels, out_channels, num_blocks,
                     expansion=6, stride=1):
    layers = [
        BottleneckBlock(
            in_channels, out_channels,
            stride=stride, expansion=expansion)]

    for _ in range(1, num_blocks):
        layers += [
            BottleneckBlock(
                out_channels, out_channels, expansion=expansion)
        ]
    return nn.Sequential(*layers)


class FeatureFusionModule(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(FeatureFusionModule, self).__init__()

        lowres_channels, highres_channels = in_channels
        self.lowres = nn.Sequential(
            ConvBnReLU2d(lowres_channels, lowres_channels,
                         kernel_size=3, padding=4, dilation=4,
                         groups=lowres_channels),
            ConvBn2d(lowres_channels, out_channels, 1)
        )
        self.highres = ConvBn2d(highres_channels, out_channels, 1)

    def forward(self, lowres, highres):
        lowres = F.interpolate(
            lowres, size=highres.shape[2:],
            mode='bilinear', align_corners=True)
        lowres = self.lowres(lowres)

        highres = self.highres(highres)

        return F.relu(lowres + highres)


class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expansion=6):
        super(BottleneckBlock, self).__init__()

        expansion_channels = in_channels * expansion
        self.conv1 = ConvBnReLU2d(in_channels, expansion_channels, 1)
        self.conv2 = ConvBnReLU2d(expansion_channels, expansion_channels, 3,
                                  padding=1, stride=stride, groups=expansion_channels)
        self.conv3 = ConvBn2d(expansion_channels, out_channels, 1)

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        if x.shape == input.shape:
            x = input + x
        return F.relu(x)
