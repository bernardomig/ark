from collections import OrderedDict
import torch
from torch import nn
from torch.nn import functional as F

from ark.nn.easy import ConvBn2d
from ark.utils.hub import register_model


@register_model(
    cityscapes={'in_channels': 3, 'out_channels': 19,
                'state_dict': 'https://files.deeplar.tk/ark-weights/espnet-cityscapes-d6791c4598.pt'},
)
def espnet(in_channels, out_channels, alphas=[2, 8]):
    return ESPNet(in_channels, out_channels, alphas=alphas)


def espnet_a(in_channels, out_channels, alphas=[2, 8]):
    return ESPNet_ABC(in_channels, out_channels, alphas=alphas, use_pyramids=False, use_connections=False)


def espnet_b(in_channels, out_channels, alphas=[2, 8]):
    return ESPNet_ABC(in_channels, out_channels, alphas=alphas, use_pyramids=False, use_connections=True)


@register_model(
    cityscapes={'in_channels': 3, 'out_channels': 19,
                'state_dict': 'https://files.deeplar.tk/ark-weights/espnet_c-cityscapes-1f76fe4247.pt'},
)
def espnet_c(in_channels, out_channels, alphas=[2, 8]):
    return ESPNet_ABC(in_channels, out_channels, alphas=alphas, use_pyramids=True, use_connections=True)


class ESPNet(nn.Module):
    def __init__(self, in_channels, out_channels, alphas=[2, 3], use_pyramids=True, use_connections=True):
        super().__init__()

        self.encoder = Encoder(in_channels, alphas=alphas,
                               use_pyramids=use_pyramids,
                               use_connections=use_connections)

        level_channels = [
            16 + in_channels if use_pyramids else 16,
            128 + in_channels if use_pyramids else 128 if use_connections else 64,
            256 if use_connections else 128,
        ]
        self.decoder = Decoder(level_channels, out_channels)

    def forward(self, input):
        outputs = self.encoder(input)
        return self.decoder(outputs)


class ESPNet_ABC(nn.Module):
    def __init__(self, in_channels, out_channels, alphas=[2, 3], use_pyramids=True, use_connections=True):
        super().__init__()

        self.encoder = Encoder(in_channels, alphas=alphas,
                               use_pyramids=use_pyramids, use_connections=use_connections,
                               return_pyramid=False)
        self.classifier = nn.Conv2d(256 if use_connections else 128, out_channels, 1, bias=False)

    def forward(self, input):
        x = self.encoder(input)
        x = self.classifier(x)
        return F.interpolate(x, size=input.shape[2:], mode='bilinear', align_corners=True)


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.level1 = ConvBnPReLU2d(in_channels[0], out_channels, 1)
        self.level2 = ConvBnPReLU2d(in_channels[1], out_channels, 1)
        self.level3 = ConvBnPReLU2d(in_channels[2], out_channels, 1)

        self.up1 = nn.ConvTranspose2d(out_channels, out_channels, 2, stride=2)
        self.up2 = DeConvBnPReLU2d(out_channels, out_channels, 2, stride=2)
        self.up3 = DeConvBnPReLU2d(out_channels, out_channels, 2, stride=2)

        self.conv1 = ConvBnPReLU2d(out_channels * 2, out_channels, 1)
        self.conv2 = ESPBlock(out_channels * 2, out_channels, use_residual=False)

    def forward(self, inputs):
        level1, level2, level3 = inputs

        level1 = self.level1(level1)
        level2 = self.level2(level2)
        level3 = self.level3(level3)

        x = torch.cat([self.up3(level3), level2], dim=1)
        x = self.conv2(x)
        x = torch.cat([self.up2(x), level1], dim=1)
        x = self.conv1(x)
        return self.up1(x)


class Encoder(nn.Module):
    def __init__(self, in_channels, alphas=[2, 3], use_pyramids=True, use_connections=True, return_pyramid=True):
        super().__init__()

        self.use_pyramids = use_pyramids
        self.use_connections = use_connections
        self.return_pyramid = return_pyramid
        if use_pyramids:
            assert use_connections

        def make_block(channels, num_blocks):
            layers = [ESPBlock(channels, channels) for _ in range(num_blocks)]
            return nn.Sequential(*layers)

        self.conv1 = ConvBnPReLU2d(in_channels, 16, 3, padding=1, stride=2)
        # if use_pyramids:
        #    self.bn1 = BNReLU(16 + in_channels)

        self.down2 = ESPBlock(in_channels + 16 if use_pyramids else 16, 64, stride=2, use_residual=False)
        self.level2 = make_block(64, alphas[0])
        # if use_pyramids:
        #     self.bn2 = BNReLU(128 + in_channels)

        self.down3 = ESPBlock(in_channels + 128 if use_pyramids else 128 if use_connections else 64,
                              128, stride=2, use_residual=False)
        self.level3 = make_block(128, alphas[1])
        # if use_pyramids:
        #     self.bn3 = BNReLU(256)

    def forward(self, input):
        if self.use_pyramids:
            half = F.avg_pool2d(input, 3, stride=2, padding=1)
            quarter = F.avg_pool2d(half, 3, stride=2, padding=1)

        x = self.conv1(input)
        x = level1 = torch.cat([x, half], dim=1) if self.use_pyramids else x
        # x = level1 = self.bn1(x) if self.use_pyramids else x

        x = x0 = self.down2(x)
        x = self.level2(x)
        x = level2 = (
            torch.cat([x, x0, quarter], dim=1) if self.use_pyramids
            else torch.cat([x, x0], dim=1) if self.use_connections
            else x
        )
        # x = level2 = self.bn2(x) if self.use_pyramids else x

        x = x0 = self.down3(x)
        x = self.level3(x)
        x = level3 = torch.cat([x0, x], dim=1) if self.use_connections else x
        # x = level3 = self.bn3(x) if self.use_pyramids else x

        return (level1, level2, level3) if self.return_pyramid else x


class ESPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilations=[1, 2, 4, 8, 16], stride=1, use_residual=True):
        super().__init__()

        self.use_residual = use_residual

        groups = len(dilations)
        mid_channels = out_channels // groups
        first_group = out_channels - mid_channels * (groups - 1)

        self.conv = ConvBnPReLU2d(in_channels, mid_channels, 3 if stride == 2 else 1,
                                  padding=1 if stride == 2 else 0,
                                  stride=stride)
        self.groups = nn.ModuleList([
            nn.Conv2d(mid_channels, mid_channels if idx != 0 else first_group, 3,
                      padding=dilation, dilation=dilation, bias=False)
            for idx, dilation in enumerate(dilations)
        ])

        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.PReLU(out_channels)

    def forward(self, input):
        x = self.conv(input)
        groups = [group(x) for group in self.groups]
        x = hierarchical_fusion(groups)
        if self.use_residual:
            x = input + x
        x = self.bn(x)
        return self.activation(x)


def hierarchical_fusion(groups):
    # the first group does not have the same number of channels as the last ones
    first, *last = groups
    add = last[0]
    adds = [first, add]

    for group in last[1:]:
        add = add + group
        adds.append(add)
    return torch.cat(adds, dim=1)


class ConvBnPReLU2d(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size,
                 padding=0,
                 stride=1,
                 dilation=1,
                 groups=1):
        super(ConvBnPReLU2d, self).__init__(OrderedDict([
            ('conv', nn.Conv2d(in_channels, out_channels, kernel_size,
                               padding=padding,
                               stride=stride,
                               dilation=dilation,
                               groups=groups,
                               bias=False)),
            ('bn', nn.BatchNorm2d(out_channels)),
            ('relu', nn.PReLU(out_channels)),
        ]))


class DeConvBnPReLU2d(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size,
                 padding=0,
                 output_padding=0,
                 stride=1,
                 dilation=1,
                 groups=1):
        super(DeConvBnPReLU2d, self).__init__(OrderedDict([
            ('conv', nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                        padding=padding,
                                        output_padding=output_padding,
                                        stride=stride,
                                        dilation=dilation,
                                        groups=groups,
                                        bias=False)),
            ('bn', nn.BatchNorm2d(out_channels)),
            ('relu', nn.PReLU(out_channels)),
        ]))
