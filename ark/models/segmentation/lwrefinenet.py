import torch
from torch import nn
from torch.nn import functional as F

from ark.nn.utils import FeatureExtractor
from ark.utils.hub import register_model


@register_model(
    voc={'in_channels': 3, 'out_channels': 21, 'state_dict': 'lwrefinenet_resnet50-voc-9aabba4cec.pt'},
)
def lwrefinenet_resnet50(in_channels, out_channels):
    from ark.models.classification.resnet import resnet50
    features = resnet50(in_channels, 1).features
    layers = (features.layer1, features.layer2, features.layer3, features.layer4)
    features = FeatureExtractor(features, layers)
    return LWRefineNet(features, out_channels)


@register_model(
    voc={'in_channels': 3, 'out_channels': 21, 'state_dict': 'lwrefinenet_resnet101-voc-4a23c15263.pt'},
    context={'in_channels': 3, 'out_channels': 60, 'state_dict': 'lwrefinenet_resnet101-context-5b8026ddec.pt'},
)
def lwrefinenet_resnet101(in_channels, out_channels):
    from ark.models.classification.resnet import resnet101
    features = resnet101(in_channels, 1).features
    layers = (features.layer1, features.layer2, features.layer3, features.layer4)
    features = FeatureExtractor(features, layers)
    return LWRefineNet(features, out_channels)


@register_model(
    voc={'in_channels': 3, 'out_channels': 21, 'state_dict': 'lwrefinenet_resnet152-voc-f2d76c1bfe.pt'},
    context={'in_channels': 3, 'out_channels': 60, 'state_dict': 'lwrefinenet_resnet152-context-229f7efdef.pt'},
)
def lwrefinenet_resnet152(in_channels, out_channels):
    from ark.models.classification.resnet import resnet152
    features = resnet152(in_channels, 1).features
    layers = (features.layer1, features.layer2, features.layer3, features.layer4)
    features = FeatureExtractor(features, layers)
    return LWRefineNet(features, out_channels)


class LWRefineNet(nn.Module):
    def __init__(self,
                 feature_extractor,
                 out_channels,
                 feature_channels=(256, 512, 1024, 2048),
                 mid_channels=((256, 256), (256, 256), (256, 256), (512,)),
                 final_channels=256,
                 crp_blocks=(4, 4, 4, 4),
                 ):
        super().__init__()

        self.features = feature_extractor

        self.crps = nn.ModuleList([
            ContextResidualPooling(channels[-1], channels[-1], num_blocks)
            for channels, num_blocks in zip(mid_channels, crp_blocks)
        ])

        self.rcus1 = nn.ModuleList([
            nn.Sequential(*[
                nn.Conv2d(in_channels if i == 0 else c, c, 1, bias=False)
                for i, c in enumerate(out_channels)
            ])
            for in_channels, out_channels in zip(feature_channels, mid_channels)
        ])

        self.rcus2 = nn.ModuleList([
            nn.Conv2d(channels[-1], final_channels, 1, bias=False)
            if idx != 0 else nn.Identity()
            for idx, channels in enumerate(mid_channels)
        ])

        self.classifier = nn.Conv2d(final_channels, out_channels, 3, padding=1)

    def forward(self, input):
        xs = self.features(input)

        for idx in reversed(range(len(xs))):
            x_ = self.rcus1[idx](xs[idx])
            if idx == len(xs) - 1:
                x = x_
            else:
                x = F.interpolate(x, x_.shape[2:], mode='bilinear', align_corners=True)
                x = x + x_
            x = torch.relu_(x)
            x = self.crps[idx](x)
            x = self.rcus2[idx](x)

        x = self.classifier(x)
        return F.interpolate(x, input.shape[2:], mode='bilinear', align_corners=True)


class ContextResidualPooling(nn.ModuleList):
    def __init__(self, in_channels, out_channels, num_blocks):
        assert in_channels == out_channels

        super().__init__([
            nn.Conv2d(out_channels, out_channels, 1, bias=False)
            for _ in range(num_blocks)
        ])

    def forward(self, input):
        x = top = input
        for conv in self.children():
            top = F.max_pool2d(top, kernel_size=5, stride=1, padding=2)
            top = conv(top)
            x = top + x
        return x
