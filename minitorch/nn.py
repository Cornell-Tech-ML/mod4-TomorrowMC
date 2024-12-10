from typing import Tuple

from .autodiff import Context
from .tensor import Tensor
from .tensor_functions import Function, rand

from typing import Optional


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0

    new_height = height // kh
    new_width = width // kw

    # Reshape the input tensor
    output = input.contiguous().view(batch, channel, new_height, kh, new_width, kw)
    output = output.permute(0, 1, 2, 4, 3, 5)
    output = output.contiguous().view(batch, channel, new_height, new_width, kh * kw)
    return output, new_height, new_width


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled average pooling 2D

    Args:
    ----
        input: Tensor of size batch * channel * height * width
        kernel: height * width of pooling

    Returns:
    -------
        Tensor of size batch * channel * new_height * new_width

    """
    output, new_height, new_width = tile(input, kernel)
    return (
        output.mean(4)
        .contiguous()
        .view(output.shape[0], output.shape[1], new_height, new_width)
    )


class Max(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        """Max forward"""
        b = a.f.max_reduce(a, int(dim.item()))
        ctx.save_for_backward(a.f.eq_zip(a, b))
        return b

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Max backward"""
        (a,) = ctx.saved_values
        return a * grad_output, 0.0


def max(input: Tensor, dim: Optional[int]) -> Tensor:
    """Apply max reduction

    Args:
    ----
        input: input Tensor
        dim: dimension to reduce

    Returns:
    -------
        Tensor with dimension dim reduced to 1

    """
    if dim is None:
        return Max.apply(input.contiguous().view(input.size), input._ensure_tensor(0))
    else:
        return Max.apply(input, input._ensure_tensor(dim))


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled max pooling 2D

    Args:
    ----
        input: Tensor of size batch * channel * height * width
        kernel: height * width of pooling

    Returns:
    -------
        Tensor of size batch * channel * new_height * new_width

    """
    output, new_height, new_width = tile(input, kernel)
    return (
        max(output, 4)
        .contiguous()
        .view(output.shape[0], output.shape[1], new_height, new_width)
    )


def softmax(input: Tensor, dim: int) -> Tensor:
    """Compute softmax as a tensor

    Args:
    ----
        input: input Tensor
        dim: dimension to compute softmax

    Returns:
    -------
        Tensor with softmax applied to dimension dim

    """
    out = input.exp()
    return out / (out.sum(dim))


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Compute log of the softmax as a tensor

    Args:
    ----
        input: input Tensor
        dim: dimension to compute log softmax

    Returns:
    -------
        Tensor with log softmax applied to dimension dim

    """
    max_tensor = max(input, dim)
    sum_tensor = (input - max_tensor).exp().sum(dim).log() + max_tensor
    return input - sum_tensor


def dropout(input: Tensor, p: float, ignore: bool = False) -> Tensor:
    """Dropout positions based on random noise, include an argument to turn off

    Args:
    ----
        input: input tensor
        p: probability of dropout
        ignore: ignore dropout if True

    Returns:
    -------
        Tensor with dropout applied

    """
    if ignore:
        return input

    if p == 0.0:
        return input

    if p == 1.0:
        return input.zeros()

    return input * (rand(input.shape, backend=input.backend, requires_grad=False) > p)