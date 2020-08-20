from typing import Tuple
from functools import partial
import torch
from torch import nn
from torch.nn import functional as F

from ark.nn.easy import ConvBnReLU2d, ConvBn2d
from ark.utils.hub import register_model

from .resnet import ResNet


__all__ = ['ResNest', 'resnest50', 'resnest101', 'resnest200', 'resnest269']


@register_model(
    imagenet1k={'in_channels': 3, 'num_classes': 1000,
                'state_dict': 'https://files.deeplar.tk/ark-weights/resnest50-imagenet-47459d8454.pt'},
)
def resnest50(in_channels, num_classes):
    r"""ResNest50

    See :class:`~ark.models.classification.resnest.ResNest` for details.
    """
    return ResNest(in_channels, num_classes,
                   block_depth=[3, 4, 6, 3],
                   init_channels=64,
                   block_channels=[256, 512, 1024, 2048],
                   radix=2,
                   )


@register_model(
    imagenet1k={'in_channels': 3, 'num_classes': 1000,
                'state_dict': 'https://files.deeplar.tk/ark-weights/resnest101-imagenet-7f4e6529c9.pt'},
)
def resnest101(in_channels, num_classes):
    r"""ResNest101

    See :class:`~ark.models.classification.resnest.ResNest` for details.
    """
    return ResNest(in_channels, num_classes,
                   block_depth=[3, 4, 23, 3],
                   init_channels=128,
                   block_channels=[256, 512, 1024, 2048],
                   radix=2,
                   )


@register_model(
    imagenet1k={'in_channels': 3, 'num_classes': 1000,
                'state_dict': 'https://files.deeplar.tk/ark-weights/resnest200-imagenet-408422922a.pt'},
)
def resnest200(in_channels, num_classes):
    r"""ResNest200

    See :class:`~ark.models.classification.resnest.ResNest` for details.
    """
    return ResNest(in_channels, num_classes,
                   block_depth=[3, 24, 36, 3],
                   init_channels=128,
                   block_channels=[256, 512, 1024, 2048],
                   radix=2,
                   )


@register_model(
    imagenet1k={'in_channels': 3, 'num_classes': 1000,
                'state_dict': 'https://files.deeplar.tk/ark-weights/resnest269-imagenet-3183fb2545.pt'},
)
def resnest269(in_channels, num_classes):
    r"""ResNest269

    See :class:`~ark.models.classification.resnest.ResNest` for details.
    """
    return ResNest(in_channels, num_classes,
                   block_depth=[3, 30, 48, 8],
                   init_channels=128,
                   block_channels=[256, 512, 1024, 2048],
                   radix=2,
                   )


class ResNest(ResNet):
    def __init__(self, in_channels: int, num_classes: int,
                 block_depth: Tuple[int, int, int, int],
                 init_channels: int,
                 block_channels: Tuple[int, int, int, int],
                 expansion: int = 4,
                 base_width: int = 64,
                 cardinality: int = 1,
                 radix: int = 1):

        assert radix != 1, "ResNest does not support radix = 1"

        block = partial(Bottleneck,
                        expansion=expansion,
                        base_width=base_width,
                        cardinality=cardinality,
                        radix=radix)

        super(ResNest, self).__init__(
            in_channels, num_classes,
            block=block,
            block_depth=block_depth,
            init_channels=init_channels,
            block_channels=block_channels)

        # change the stem to match the ResNest architecture
        self.features.stem = nn.Sequential(
            ConvBnReLU2d(in_channels, init_channels // 2, 3, padding=1, stride=2),
            ConvBnReLU2d(init_channels // 2, init_channels // 2, 3, padding=1),
            ConvBnReLU2d(init_channels // 2, init_channels, 3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )


class Bottleneck(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 stride: int = 1,
                 expansion: int = 4,
                 base_width: int = 64,
                 cardinality: int = 1,
                 radix: int = 1):
        super(Bottleneck, self).__init__()

        self.radix = radix

        width = int((out_channels / expansion) * (base_width / 64) * cardinality)

        self.conv1 = ConvBnReLU2d(in_channels, width, 1)

        self.conv2 = ConvBnReLU2d(width, width * radix, 3, padding=1, groups=cardinality * radix)
        self.sa = SplitAttention(width, width * radix, 4, cardinality=cardinality, radix=radix)
        self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1) if stride == 2 else nn.Identity()
        self.conv3 = ConvBn2d(width, out_channels, 1)

        self.activation = nn.ReLU(inplace=True)

        self.downsample = (
            nn.Sequential(
                nn.AvgPool2d(stride) if stride != 1 else nn.Identity(),
                ConvBn2d(in_channels, out_channels, 1),
            )
            if stride != 1 or in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.sa(x)
        x = self.pool(x)
        x = self.conv3(x)
        residual = self.downsample(input)
        return self.activation(x + residual)


class SplitAttention(nn.Module):
    cardinality: int
    radix: int

    def __init__(self, in_channels, out_channels, reduction, cardinality, radix):
        super(SplitAttention, self).__init__()

        self.cardinality = cardinality
        self.radix = radix

        mid_channels = out_channels // reduction

        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, groups=cardinality)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, 1, groups=cardinality)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        xs = torch.chunk(input, self.radix, dim=1)
        x = sum(xs)
        at = F.adaptive_avg_pool2d(x, 1)
        at = self.conv1(at)
        at = self.bn1(at)
        at = self.relu(at)
        at = self.conv2(at)
        at = radix_softmax(at, self.radix, self.cardinality)

        ats = torch.split(at, x.size(1), dim=1)

        return sum((at * x for at, x in zip(ats, xs)))


@torch.jit.script
def radix_softmax(input: torch.Tensor, radix: int, cardinality: int):
    batch = input.size(0)
    x = input.view(batch, cardinality, radix, -1).transpose(1, 2)
    x = torch.softmax(x, dim=1)
    return x.reshape_as(input)


if __name__ == "__main__":
    x = torch.randn((3, 3, 128, 128))
    model = resnest50(3, 1000)
    y = model(x)
