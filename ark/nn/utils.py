from typing import Union, List, Callable, Optional, Tuple
from contextlib import ExitStack
from torch import nn


def round_channels(channels: int, divisor: int = 8):
    r"""Rounds a number to the nearest multiple of the divisor. 

    Useful to ensure that number of channels is GPU friendly 
    (multiple of 8 or 4), in networks scaled by a width multiplier.
    """
    c = int(channels + divisor / 2) // divisor * divisor
    c = c + divisor if c < (0.9 * channels) else c
    return c


class IntermediateGrabber:
    r"""A context-based class for capturing the intermediate input/output at certain layers 
    in a module. This class is specially useful for debugging or as construction block for 
    difficult skip-connection models (see :class:`~ark.nn.utils.FeatureExtractor`).

    This class is used inside a `with` block, where it registers a forward hook and, when 
    exiting, clears the hook so that the underlying layer is unmodified. The extracted 
    input/output will be available as the `.value` property.

    Args:
        layer (Module): the target layer at which the input/output will be saved
        type (str): determines if the feature will be saved at the input or the output of 
            the layer

    Example::
        >>> model = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
        >>> x = torch.randn((3, 3, 32, 32))
        >>> with IntermediateGrabber(model[2]) as g:
                y = model(x)
        >>> print(g.value.shape)
        torch.Size([3, 16, 16, 16])

    """

    def __init__(self,
                 layer: nn.Module,
                 type: int = "output"):
        assert type in {'input', 'output'}, \
            "type of IntermediateGrabber must be either input or output"

        self.layer = layer
        self.type = type

    def __enter__(self):
        def grabber_hook(_module, input, output):
            self.value = input if self.type == 'input' else output

        self._hook = self.layer.register_forward_hook(grabber_hook)

        return self

    def __exit__(self, _type, _value, _tb):
        self._hook.remove()
        del self._hook


class FeatureExtractor(nn.Module):
    r"""Wraps a existing :obj:`~nn.Module` to extract the features at intermediate layers. 
    This class is specially useful for models that repurpose the classification models' feature 
    encoders (i.e. ResNets or MobileNets) as backbones and fuse the intermediate features at 
    certain intermediate layers.

    Args:
        module (Module): the object that will be wrapped by this class. This class will behave
            exactly like this object (except the forward function)
        layers (list of Module): the intermediate layers to extract the features. For added flexibility, 
            wrap the layers as :class:`~ark.nn.utils.IntermediateGrabber`
        include_output (bool): return the output of the module as along with the intermediate features. 
            If so, the output will be a tuple of (output, intermediates). Default: false

    Example::

        >>> from ark.models.classification.resnet import resnet50
        >>> model = resnet50(pretrained='imagenet1k')
        >>> features = model.features
        >>> layers = (features.layer2, features.layer3, features.layer4)
        >>> backbone = FeatureExtractor(features, layers)
        >>> x = torch.randn((3, 3, 64, 64))
        >>> xs = backbone(x)

    """
    include_output: bool

    def __init__(self,
                 module: nn.Module,
                 layers: List[Union[IntermediateGrabber, nn.Module]],
                 include_output: bool = False):
        # This class works as an object wrapper
        # So we cannot call the super constructor:
        # super().__init__()
        # Instead, we will override the base class of this method, to match the
        # class of the module its properties through meta-programming
        self.__class__ = type(module.__class__.__name__,
                              (self.__class__, module.__class__),
                              {})
        self.__dict__ = module.__dict__

        self._intermediate_layers = [
            layer if isinstance(layer, IntermediateGrabber) else IntermediateGrabber(layer)
            for layer in layers
        ]
        self.include_output = include_output

    def forward(self, input):
        with ExitStack() as stack:
            for layer in self._intermediate_layers:
                stack.enter_context(layer)

            output = super().forward(input)

        intermediates = [layer.value for layer in self._intermediate_layers]

        if self.include_output:
            return output, intermediates
        else:
            return intermediates


class DeepSupervisor(nn.Module):
    def __init__(self,
                 module: nn.Module,
                 layers: List[Tuple[nn.Module, nn.Module]]):
        super(DeepSupervisor, self).__init__()

        self.module = module
        self._supervised_layers = [
            layer if isinstance(layer, IntermediateGrabber) else IntermediateGrabber(layer)
            for layer, _ in layers
        ]
        self.auxiliaries = nn.ModuleList([
            auxiliary for _, auxiliary in layers
        ])

    def forward(self, input):
        if self.training:
            with ExitStack() as stack:
                for layer in self._supervised_layers:
                    stack.enter_context(layer)

                output = self.module(input)

            aux_outputs = [
                aux(layer.value) for layer, aux in zip(self._supervised_layers, self.auxiliaries)
            ]

            return output, aux_outputs
        else:
            return self.module(input)
