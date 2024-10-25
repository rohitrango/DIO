import torch

def v2img_2d(velocity: torch.Tensor):
    ''' convert [B, H, W, chan] to [B, chan, H, W] '''
    return velocity.permute(0, 3, 1, 2)

def img2v_2d(image: torch.Tensor):
    ''' convert [B, chan, H, W] to [B, H, W, chan] '''
    return image.permute(0, 2, 3, 1)

def v2img_3d(velocity: torch.Tensor):
    ''' convert [B, D, H, W, chan] to [B, chan, D, H, W] '''
    return velocity.permute(0, 4, 1, 2, 3)

def img2v_3d(image: torch.Tensor):
    ''' convert [B, chan, D, H, W] to [B, D, H, W, chan] '''
    return image.permute(0, 2, 3, 4, 1)

from time import time, sleep
import torch
from torch.utils.checkpoint import checkpoint
from torch import nn
from torch.nn import functional as F
from typing import Union, Tuple, List, Optional, Dict, Any, Callable, TypeVar
# Use item or list of items
T = TypeVar("T")
ItemOrList = Union[T, List[T]]

@torch.jit.script
def gaussian_1d(
    sigma: torch.Tensor, truncated: float = 4.0, approx: str = "erf", normalize: bool = True
) -> torch.Tensor:
    """
    one dimensional Gaussian kernel.
    Args:
        sigma: std of the kernel
        truncated: tail length
        approx: discrete Gaussian kernel type, available options are "erf", "sampled", and "scalespace".
            - ``erf`` approximation interpolates the error function;
            - ``sampled`` uses a sampled Gaussian kernel;
            - ``scalespace`` corresponds to
              https://en.wikipedia.org/wiki/Scale_space_implementation#The_discrete_Gaussian_kernel
              based on the modified Bessel functions.
        normalize: whether to normalize the kernel with `kernel.sum()`.
    Raises:
        ValueError: When ``truncated`` is non-positive.
    Returns:
        1D torch tensor
    """
    sigma = torch.as_tensor(sigma, dtype=torch.float, device=sigma.device if isinstance(sigma, torch.Tensor) else None)
    device = sigma.device
    if truncated <= 0.0:
        raise ValueError(f"truncated must be positive, got {truncated}.")
    tail = int(max(float(sigma) * truncated, 0.5) + 0.5)
    if approx.lower() == "erf":
        x = torch.arange(-tail, tail + 1, dtype=torch.float, device=device)
        t = 0.70710678 / torch.abs(sigma)
        out = 0.5 * ((t * (x + 0.5)).erf() - (t * (x - 0.5)).erf())
        out = out.clamp(min=0)
    elif approx.lower() == "sampled":
        x = torch.arange(-tail, tail + 1, dtype=torch.float, device=sigma.device)
        out = torch.exp(-0.5 / (sigma * sigma) * x**2)
        if not normalize:  # compute the normalizer
            out = out / (2.5066282 * sigma)
    else:
        raise NotImplementedError(f"Unsupported option: approx='{approx}'.")
    return out / out.sum() if normalize else out  # type: ignore


@torch.jit.script
def make_rectangular_kernel(kernel_size: int) -> torch.Tensor:
    return torch.ones(kernel_size)

@torch.jit.script
def make_triangular_kernel(kernel_size: int) -> torch.Tensor:
    fsize = (kernel_size + 1) // 2
    if fsize % 2 == 0:
        fsize -= 1
    f = torch.ones((1, 1, fsize), dtype=torch.float).div(fsize)
    padding = (kernel_size - fsize) // 2 + fsize // 2
    return F.conv1d(f, f, padding=padding).reshape(-1)

@torch.jit.script
def make_gaussian_kernel(kernel_size: int) -> torch.Tensor:
    sigma = torch.tensor(kernel_size / 3.0)
    kernel = gaussian_1d(sigma=sigma, truncated=(kernel_size // 2) * 1.0, approx="sampled", normalize=False) * (
        2.5066282 * sigma
    )
    return kernel[:kernel_size]

@torch.jit.script
def _separable_filtering_conv(
    input_: torch.Tensor,
    kernels: List[torch.Tensor],
    pad_mode: str,
    spatial_dims: int,
    paddings: List[int],
    num_channels: int,
) -> torch.Tensor:

    # re-write from recursive to non-recursive for torch.jit to work
    # for d in range(spatial_dims-1, -1, -1):
    for d in range(spatial_dims):
        s = [1] * len(input_.shape)
        s[d + 2] = -1
        _kernel = kernels[d].reshape(s)
        # if filter kernel is unity, don't convolve
        if _kernel.numel() == 1 and _kernel[0] == 1:
            continue

        _kernel = _kernel.repeat([num_channels, 1] + [1] * spatial_dims)
        _padding = [0] * spatial_dims
        _padding[d] = paddings[d]
        _reversed_padding = _padding[::-1]

        # translate padding for input to torch.nn.functional.pad
        _reversed_padding_repeated_twice: list[list[int]] = [[p, p] for p in _reversed_padding]
        _sum_reversed_padding_repeated_twice: list[int] = []
        for p in _reversed_padding_repeated_twice:
            _sum_reversed_padding_repeated_twice.extend(p)
        # _sum_reversed_padding_repeated_twice: list[int] = sum(_reversed_padding_repeated_twice, [])

        padded_input = F.pad(input_, _sum_reversed_padding_repeated_twice, mode=pad_mode)
        # update input
        if spatial_dims == 1:
            input_ = F.conv1d(input=padded_input, weight=_kernel, groups=num_channels)
        elif spatial_dims == 2:
            input_ = F.conv2d(input=padded_input, weight=_kernel, groups=num_channels)
        elif spatial_dims == 3:
            input_ = F.conv3d(input=padded_input, weight=_kernel, groups=num_channels)
        else:
            raise NotImplementedError(f"Unsupported spatial_dims: {spatial_dims}.")
    return input_

# @torch.jit.script
def separable_filtering(x: torch.Tensor, kernels: torch.Tensor, mode: str = "zeros") -> torch.Tensor:
    """
    Apply 1-D convolutions along each spatial dimension of `x`.
    Args:
        x: the input image. must have shape (batch, channels, H[, W, ...]).
        kernels: kernel along each spatial dimension.
            could be a single kernel (duplicated for all spatial dimensions), or
            a list of `spatial_dims` number of kernels.
        mode (string, optional): padding mode passed to convolution class. ``'zeros'``, ``'reflect'``, ``'replicate'``
            or ``'circular'``. Default: ``'zeros'``. See ``torch.nn.Conv1d()`` for more information.
    Raises:
        TypeError: When ``x`` is not a ``torch.Tensor``.
    Examples:
    .. code-block:: python
        >>> import torch
        >>> img = torch.randn(2, 4, 32, 32)  # batch_size 2, channels 4, 32x32 2D images
        # applying a [-1, 0, 1] filter along each of the spatial dimensions.
        # the output shape is the same as the input shape.
        >>> out = separable_filtering(img, torch.tensor((-1., 0., 1.)))
        # applying `[-1, 0, 1]`, `[1, 0, -1]` filters along two spatial dimensions respectively.
        # the output shape is the same as the input shape.
        >>> out = separable_filtering(img, [torch.tensor((-1., 0., 1.)), torch.tensor((1., 0., -1.))])
    """

    if not isinstance(x, torch.Tensor):
        raise TypeError(f"x must be a torch.Tensor but is {type(x).__name__}.")

    spatial_dims = len(x.shape) - 2
    if isinstance(kernels, torch.Tensor):
        kernels = [kernels] * spatial_dims
    _kernels = [s.to(x) for s in kernels]
    _paddings = [(k.shape[0] - 1) // 2 for k in _kernels]
    n_chs = x.shape[1]
    pad_mode = "constant" if mode == "zeros" else mode
    return _separable_filtering_conv(x, _kernels, pad_mode, spatial_dims, _paddings, n_chs)
