from collections import OrderedDict
import torch
from torch import nn

from ark.nn.easy import ConvBnReLU2d

def densenet121(in_channels, num_classes):
    return DenseNet(in_channels, num_classes,
        growth=32,
        block_depth=[6, 12 ,24, 16],
        init_features=64)

def densenet161(in_channels, num_classes):
    return DenseNet(in_channels, num_classes,
        growth=48,
        block_depth=[6, 12, 36, 24],
        init_features=96)

def densenet169(in_channels, num_classes):
    return DenseNet(in_channels, num_classes,
        growth=32, 
        block_depth=[6,12, 32, 32],
        init_features=64)
    
def densenet201(in_channels, num_classes):
    return DenseNet(in_channels, num_classes,
        growth=32,
        block_depth=[6, 12, 48, 32],
        init_features=64)

class DenseNet(nn.Sequential):
    def __init__(self, in_channels, num_classes, 
                growth=32, 
                block_depth=[6, 12, 24, 16], 
                init_features=64, 
                expansion=4, 
                dropout_p=0.0):

        stem = nn.Sequential(OrderedDict([
            ('conv', ConvBnReLU2d(in_channels, init_features, 7, padding=3, stride=2)),
            ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        layers = [('stem', stem)]
        channels = init_features
        for idx, depth in enumerate(block_depth):
            out_channels = channels + growth * depth
            layer = [
                DenseBlock(channels + growth * idx, channels + growth * (idx+1), 
                           growth=growth, 
                           expansion=expansion, 
                           dropout_p=dropout_p)
                for idx in range(depth)
            ]
            if idx != len(block_depth) - 1:
                # if it is not the last block, add the transition block
                layer += TransitionBlock(out_channels, out_channels // 2)
                out_channels = out_channels // 2

            layers += [(f'layer{idx+1}', nn.Sequential(*layer))]
            channels = out_channels
        layers += [('tail', nn.Sequential(nn.BatchNorm2d(channels), nn.ReLU(inplace=True)))]

        features = nn.Sequential(OrderedDict(layers))

        classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, num_classes)
        )


        super(DenseNet, self).__init__(OrderedDict([
            ('features', features),
            ('classifier', classifier),
        ]))

class DenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, growth, expansion, dropout_p):
        super(DenseBlock, self).__init__()

        assert in_channels + growth == out_channels

        self.conv1 = BnReLUConv2d(in_channels, growth * expansion, 1)
        self.conv2 = BnReLUConv2d(growth * expansion, growth, 3, padding=1)

        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.dropout(x)
        return torch.cat([input, x], dim=1)

class TransitionBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(TransitionBlock, self).__init__(OrderedDict([
            ('conv', BnReLUConv2d(in_channels, out_channels, 1)),
            ('pool', nn.AvgPool2d(2)),
        ]))

    


class BnReLUConv2d(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size,
                 padding=0,
                 stride=1,
                 dilation=1,
                 groups=1):
        super(BnReLUConv2d, self).__init__(OrderedDict([
            ('bn', nn.BatchNorm2d(in_channels)),
            ('relu', nn.ReLU(inplace=True)),
            ('conv', nn.Conv2d(in_channels, out_channels, kernel_size,
                               padding=padding,
                               stride=stride,
                               dilation=dilation,
                               groups=groups,
                               bias=False)),
        ]))