
from math import log
from typing import Union, Optional
import torch
from torch import Tensor
from torch.nn.modules.loss import _WeightedLoss
from torch.nn import functional as F


def ohem_cross_entropy(input: Tensor, target: Tensor,
                       weight: Optional[Tensor] = None,
                       ignore_index: int = -100,
                       threshold: float = -log(0.7),
                       num_frac: float = 0.08):
    r"""Implements the Online Hard Example Mining Cross Entropy loss.

    See :class:`ark.nn.OHEMCrossEntropyLoss` for details.
    """

    numel = torch.numel(target[target != ignore_index])
    n_min = int(numel * num_frac)

    loss = F.cross_entropy(input, target, ignore_index=ignore_index, weight=weight, reduction='none')
    loss = loss.flatten()
    hard = loss[loss > threshold]

    return (loss.topk(n_min)[0] if hard.numel() < n_min else hard).mean()


class OHEMCrossEntropyLoss(_WeightedLoss):
    r"""A modified cross-entropy criterion that actively focuses on bad examples 
    and discard easy examples, for a improved training.

    Args:
        weight (Tensor, optional): a manual rescaling weight given to each class,
            used in the underlying cross-entropy loss.
            If given, has to be a tensor of size `C`
        ignore_index (int, optional): specifies a target value that is ignored, 
            used in the underlying cross-entropy loss.
        threshold (float, optional): defines a the threshold that defines a 
            hard example. All the hard examples contribute to the loss. 
            Default: :math:`-log(0.7)`
        num_frac (int, optional): defines the minimum fraction of examples that 
            contribute to the loss. Default: 0.08

    Example::

        >>> criterion = OHEMCrossEntropyLoss()
        >>> x = torch.randn((3, 4, 16, 16))
        >>> y = torch.randint(4, (3, 16, 16))
        >>> loss = criterion(x, y)

    """

    def __init__(self,
                 weight: Optional[Tensor] = None,
                 ignore_index: int = -100,
                 threshold: float = -log(0.7),
                 num_frac: float = 0.08):
        super(OHEMCrossEntropyLoss, self).__init__(weight=weight)

        self.threshold = threshold
        self.ignore_index = ignore_index
        self.num_frac = num_frac

    def forward(self, input: Tensor, target: Tensor):
        return ohem_cross_entropy(input, target,
                                  weight=self.weight,
                                  ignore_index=self.ignore_index,
                                  threshold=self.threshold,
                                  num_frac=self.num_frac)
