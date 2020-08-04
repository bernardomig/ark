import torch
from torch.nn import Module

from ..functional import mish, swish


class Swish(Module):
    r"""Implements the element-wise function:

    .. math::
        \text{Swish}(x) = x * \sigma(x)

    Args:
        inplace: can optionally do the operation in-place: Default: ``False``

    Shape:
        - Input: :math:`(N, *)` where `*` means any number of additional dimensions
        - Output: :math:`(N, *)`, same as the input

    Examples::

        >>> m = Swish()
        >>> input = torch.randn((3,3))
        >>> output = m(input)

    .. _`Searching for Activation Functions`: https://arxiv.org/abs/1710.05941
    """

    def __init__(self, inplace: bool = False):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, input):
        return swish(input, inplace=self.inplace)


class Mish(Module):
    r"""Implements the element-wise function:

    .. math::
        \text{Mish}(x) = x * \tanh{\text{Softplus}(x)}

    Args:
        inplace: can optionally do the operation in-place: Default: ``False``

    Shape:
        - Input: :math:`(N, *)` where `*` means any number of additional dimensions
        - Output: :math:`(N, *)`, same as the input

    Examples::

        >>> m = Mish()
        >>> input = torch.randn((3,3))
        >>> output = m(input)

    .. _`Mish: A Self Regularized Non-Monotonic Neural Activation Function`: 
        https://arxiv.org/abs/1908.08681
    """

    def __init__(self, inplace: bool = False):
        super(Mish, self).__init__()
        self.inplace = inplace

    def forward(self, input):
        return mish(input, inplace=self.inplace)
