from collections import OrderedDict
from torch import nn
from torch.nn import functional as F

from ark.nn.easy import ConvBnReLU2d, ConvBn2d


def resnet18(in_channels, num_classes):
    return ResNet(in_channels, num_classes,
                  block_depth=[2, 2, 2, 2],
                  block=BasicBlock)


def resnet34(in_channels, num_classes):
    return ResNet(in_channels, num_classes,
                  block_depth=[3, 4, 6, 3],
                  block=BasicBlock)


def resnet50(in_channels, num_classes):
    return ResNet(in_channels, num_classes,
                  block_depth=[3, 4, 6, 3],
                  block=Bottleneck,
                  expansion=4)


def resnet101(in_channels, num_classes):
    return ResNet(in_channels, num_classes,
                  block_depth=[3, 4, 23, 3],
                  block=Bottleneck,
                  expansion=4)


def resnet152(in_channels, num_classes):
    return ResNet(in_channels, num_classes,
                  block_depth=[3, 8, 36, 3],
                  block=Bottleneck,
                  expansion=4)


class ResNet(nn.Sequential):
    def __init__(self, in_channels, num_classes,
                 block_depth, block, expansion=1):

        def make_layer(in_channels, out_channels, num_blocks, stride=2):
            layers = [block(in_channels, out_channels, stride=stride)]
            for _ in range(1, num_blocks):
                layers += [block(out_channels, out_channels)]
            return nn.Sequential(*layers)

        e = expansion

        features = nn.Sequential(OrderedDict([
            ('stem', nn.Sequential(OrderedDict([
                ('conv', ConvBnReLU2d(in_channels, 64, 7, stride=2, padding=3)),
                ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
            ]))),
            ('layer1', make_layer(64, e * 64, block_depth[0], stride=1)),
            ('layer2', make_layer(e * 64, e * 128, block_depth[1])),
            ('layer3', make_layer(e * 128, e*256, block_depth[2])),
            ('layer4', make_layer(e * 256, e * 512, block_depth[3])),
        ]))

        classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(e * 512, num_classes),
        )

        super().__init__(OrderedDict([
            ('features', features),
            ('classifier', classifier),
        ]))

    @staticmethod
    def replace_stride_with_dilation(model, output_stride, multigrid_rates=None):
        assert output_stride in {8, 16, 32}, \
            f"output_stride should be 8, 16, or 32. Got {output_stride}"

        def patch_layer(layer, dilation, stride=1, multigrid_rates=None):
            from torch.nn.modules.utils import _pair

            if isinstance(layer[0], BasicBlock):
                layer[0].conv1.conv.stride = _pair(stride)
            elif isinstance(layer[0], Bottleneck):
                layer[0].conv2.conv.stride = _pair(stride)
            layer[0].downsample.conv.stride = _pair(stride)

            for id, m in enumerate(layer.children()):
                rate = 1 if multigrid_rates is None else multigrid_rates[id]

                if isinstance(m, BasicBlock):
                    m.conv1.conv.dilation = _pair(rate * dilation)
                    m.conv1.conv.padding = _pair(rate * dilation)
                if isinstance(m, Bottleneck):
                    m.conv2.conv.dilation = _pair(rate * dilation)
                    m.conv2.conv.padding = _pair(rate * dilation)

        if output_stride == 8:
            patch_layer(model.features.layer3, dilation=2)
            patch_layer(model.features.layer4, dilation=4, multigrid_rates=multigrid_rates)
        elif output_stride == 16:
            patch_layer(model.features.layer4, dilation=2, multigrid_rates=multigrid_rates)
        elif output_stride == 32:
            patch_layer(model.features.layer3, dilation=1, stride=2)
            patch_layer(model.features.layer4, dilation=1, stride=2)


replace_stride_with_dilation = ResNet.replace_stride_with_dilation


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = ConvBnReLU2d(in_channels, out_channels, 3, padding=1, stride=stride)
        self.conv2 = ConvBn2d(out_channels, out_channels, 3, padding=1)

        self.downsample = (
            ConvBn2d(in_channels, out_channels, 1, stride=stride)
            if in_channels != out_channels or stride != 1
            else nn.Identity()
        )

        self.activation = nn.ReLU(inplace=True)

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        residual = self.downsample(input)
        return self.activation(x + residual)


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expansion=4):
        super(Bottleneck, self).__init__()

        width = out_channels // expansion

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
