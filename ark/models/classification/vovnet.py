from collections import OrderedDict
import torch
from torch import nn

from ark.nn.easy import ConvBnReLU2d
from ark.utils.hub import register_model


@register_model(
    imagenet1k=dict(in_channels=3, num_classes=1000, state_dict='vovnet39-imagenet1k-b39416fe8a.pt')
)
def vovnet37(in_channels, num_classes):
    return VOVNet(in_channels, num_classes, block_depth=[1, 1, 2, 2])


@register_model(
    imagenet1k=dict(in_channels=3, num_classes=1000, state_dict='vovnet57-imagenet1k-808a175499.pt')
)
def vovnet57(in_channels, num_classes):
    return VOVNet(in_channels, num_classes, block_depth=[1, 1, 4, 3])


class VOVNet(nn.Sequential):
    def __init__(self, in_channels, num_classes,
                 block_depth=[1, 1, 1, 1],
                 init_channels=128,
                 block_channels=[256, 512, 768, 1024],
                 reduction_channels=[128, 160, 192, 224],
                 osa_blocks=5,
                 ):
        super().__init__()

        def make_layer(in_channels, out_channels, num_blocks, reduction_channels, stride=2):
            layers = []
            if stride == 2:
                layers += [nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)]
            layers += [OneShotAggregation(in_channels, out_channels, reduction_channels, osa_blocks)]
            for _ in range(1, num_blocks):
                layers += [OneShotAggregation(out_channels, out_channels, reduction_channels, osa_blocks)]
            return nn.Sequential(*layers)

        stem = nn.Sequential(
            ConvBnReLU2d(in_channels, init_channels//2, 3, stride=2, padding=1),
            ConvBnReLU2d(init_channels//2, init_channels//2, 3, padding=1),
            ConvBnReLU2d(init_channels//2, init_channels, 3, stride=2, padding=1),
        )

        # for redability
        c = block_channels
        d = block_depth
        r = reduction_channels

        features = nn.Sequential(OrderedDict([
            ('stem', stem),
            ('layer1', make_layer(init_channels, c[0], d[0], r[0], stride=1)),
            ('layer2', make_layer(c[0], c[1], d[1], r[1])),
            ('layer3', make_layer(c[1], c[2], d[2], r[2])),
            ('layer4', make_layer(c[2], c[3], d[3], r[3])),
        ]))

        classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(block_channels[-1], num_classes)
        )

        super(VOVNet, self).__init__(OrderedDict([
            ('features', features),
            ('classifier', classifier),
        ]))


class OneShotAggregation(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 reduction_channels,
                 num_blocks=5):
        super().__init__()

        convs = []
        for i in range(num_blocks):
            channels = in_channels if i == 0 else reduction_channels
            convs += [ConvBnReLU2d(channels, reduction_channels, 3, padding=1)]
        self.convs = nn.ModuleList(convs)

        channels = in_channels + reduction_channels * num_blocks
        self.transition = ConvBnReLU2d(channels, out_channels, 1)

    def forward(self, input):
        x = input
        xs = [x]

        for conv in self.convs:
            x = conv(x)
            xs.append(x)

        x = torch.cat(xs, dim=1)

        x = self.transition(x)

        if input.shape == x.shape:
            x = x + input

        return x
