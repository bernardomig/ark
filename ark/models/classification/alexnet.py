from collections import OrderedDict
from torch import nn

from ark.utils.hub import register_model

__all__ = ['AlexNet', 'alexnet']


@register_model(
    imagenet1k={'in_channels': 3, 'num_classes': 1000,
                'state_dict': 'alexnet-imagenet-30d827d4ff.pt'},
)
def alexnet(in_channels, num_classes):
    r"""Alexnet model.

    See :class:`~ark.models.classification.alexnet.AlexNet` for details.
    """
    return AlexNet(in_channels=in_channels, num_classes=num_classes)


class AlexNet(nn.Sequential):
    r"""AlexNet model implementation from 
    `"ImageNet Classification with Deep Convolutional Neural Networks"
    <https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks>`_ paper.

    Args:
        in_channels (int): the input channels
        num_classes (int): the number of the output classification classes
    """

    def __init__(self, in_channels: int, num_classes: int):
        features = nn.Sequential(OrderedDict([
            ('layer1', nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(in_channels, 64, 11, stride=4, padding=2)),
                ('relu', nn.ReLU(inplace=True)),
                ('pool', nn.MaxPool2d(kernel_size=3, stride=2)),
            ]))),
            ('layer2', nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(64, 192, 5, padding=2)),
                ('relu', nn.ReLU(inplace=True)),
                ('pool', nn.MaxPool2d(kernel_size=3, stride=2)),
            ]))),
            ('layer3', nn.Sequential(OrderedDict([
                ('conv1', nn.Conv2d(192, 384, 3, padding=1)),
                ('relu1', nn.ReLU(inplace=True)),
                ('conv2', nn.Conv2d(384, 256, 3, padding=1)),
                ('relu2', nn.ReLU(inplace=True)),
                ('conv3', nn.Conv2d(256, 256, 3, padding=1)),
                ('relu3', nn.ReLU(inplace=True)),
                ('pool', nn.MaxPool2d(kernel_size=3, stride=2)),
            ]))),
        ]))

        classifier = nn.Sequential(
            nn.Dropout(),
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

        super().__init__(OrderedDict([
            ('features', features),
            ('classifier', classifier),
        ]))
