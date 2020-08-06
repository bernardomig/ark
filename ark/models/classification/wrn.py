from functools import partial
from torch import nn
from torch.nn import functional as F

from ark.nn.easy import ConvBnReLU2d, ConvBn2d
from ark.nn.utils import round_by

from .resnet import ResNet

def wrn50_2_0(in_channels, out_channels):
    return WRN(in_channels, out_channels, 
                block_depth=[3, 4, 6, 3],
                width_multiplier=2.)

def wrn101_2_0(in_channels, out_channels):
    return WRN(in_channels, out_channels, 
                block_depth=[3, 4, 23, 3],
                width_multiplier=2.)


class WRN(ResNet):
    def __init__(self, in_channels, num_classes, block_depth, expansion=4, width_multiplier=1.):
        super().__init__(in_channels, num_classes, block_depth, 
                        block=partial(Bottleneck, expansion=expansion, width_multiplier=width_multiplier), 
                        expansion=expansion)


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expansion=4, width_multiplier=1):
        super(Bottleneck, self).__init__()

        width = round_by(out_channels * width_multiplier / expansion)

        self.conv1 = ConvBnReLU2d(in_channels, width, 1)
        self.conv2 = ConvBnReLU2d(width, width, 3, padding=1, stride=stride)
        self.conv3 = ConvBn2d(width, out_channels, 1)

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
        residual = self.downsample(input)
        return self.activation(x + residual)