import torch
from torch import Tensor
from torch.nn import functional as F
from torch.autograd import Function

__all__ = ['swish', 'mish']


class SwishFunction(Function):
    @staticmethod
    def forward(ctx, x: Tensor, inplace: bool = False):
        ctx.save_for_backward(x)
        x_sigmoid = torch.sigmoid(x)
        return x.mul_(x_sigmoid) if inplace else x.mul(x_sigmoid)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_variables[0]
        x_s = torch.sigmoid(x)
        return grad_output * (x_s * (1 + x * (1 - x_s)))


def swish(input: Tensor, inplace: bool = False):
    r"""Applies the swish activation function.

    See :class:`~ark.nn.Swish` for details.
    """
    return SwishFunction.apply(input, inplace)


class MishFunction(Function):
    @staticmethod
    def forward(ctx, x: Tensor, inplace: bool = False):
        ctx.save_for_backward(x)
        x_ts = torch.tanh_(F.softplus(x))
        return x.mul_(x_ts) if inplace else x.mul(x_ts)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_variables[0]
        x_s = torch.sigmoid(x)
        x_ts = torch.tanh_(F.softplus(x))
        return grad_output * (x_ts + x * x_s * (1-x_ts * x_ts))


def mish(input: Tensor, inplace: bool = False):
    r"""Applies the mish activation function.

    See :class:`~ark.nn.Mish` for details.
    """
    return MishFunction.apply(input, inplace)
