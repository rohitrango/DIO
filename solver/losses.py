import torch
import numpy as np
import torch.nn as nn
import math
from torch.nn import functional as F
from functools import partial

'''
Cross correlation
'''
from time import time, sleep
import torch
from torch.utils.checkpoint import checkpoint
from torch import nn
from torch.nn import functional as F
from typing import Union, Tuple, List, Optional, Dict, Any, Callable
from typing import TypeVar
T = TypeVar("T")
ItemOrList = Union[T, List[T]]

# from solver.adam import ALIGN_CORNERS as 
align_corners = False

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
# def separable_filtering(x: torch.Tensor, kernels: torch.Tensor, mode: str = "zeros") -> torch.Tensor:
def separable_filtering(x, kernels, mode='zeros') -> torch.Tensor:
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

    # if not isinstance(x, torch.Tensor):
    #     raise TypeError(f"x must be a torch.Tensor but is {type(x).__name__}.")

    spatial_dims = len(x.shape) - 2
    if isinstance(kernels, torch.Tensor):
        kernels = [kernels] * spatial_dims
    _kernels = [s.to(x) for s in kernels]
    _paddings = [(k.shape[0] - 1) // 2 for k in _kernels]
    n_chs = x.shape[1]
    pad_mode = "constant" if mode == "zeros" else mode
    return _separable_filtering_conv(x, _kernels, pad_mode, spatial_dims, _paddings, n_chs)


# dict
kernel_dict = {
    "rectangular": make_rectangular_kernel,
    "triangular": make_triangular_kernel,
    "gaussian": make_gaussian_kernel,
}

class LocalNormalizedCrossCorrelationLoss(nn.Module):
    """
    Local squared zero-normalized cross-correlation.
    The loss is based on a moving kernel/window over the y_true/y_pred,
    within the window the square of zncc is calculated.
    The kernel can be a rectangular / triangular / gaussian window.
    The final loss is the averaged loss over all windows.
    Adapted from:
        https://github.com/voxelmorph/voxelmorph/blob/legacy/src/losses.py
        DeepReg (https://github.com/DeepRegNet/DeepReg)
    """

    def __init__(
        self,
        spatial_dims: int = 3,
        kernel_size: int = 3,
        kernel_type: str = "rectangular",
        reduction: str = "mean",
        smooth_nr: float = 0.0,
        smooth_dr: float = 1e-5,
        image_noise_sigma: float = 0,
        unsigned: bool = True,
        checkpointing: bool = False,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions, {``1``, ``2``, ``3``}. Defaults to 3.
            kernel_size: kernel spatial size, must be odd.
            kernel_type: {``"rectangular"``, ``"triangular"``, ``"gaussian"``}. Defaults to ``"rectangular"``.
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.
                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.
            smooth_nr: a small constant added to the numerator to avoid nan.
            smooth_dr: a small constant added to the denominator to avoid nan.
            image_noise_sigma: a small noise image to make CC return 1 instead of 0 for constant regions
            unsigned: whether to disregard the sign of the cross correlation (True by default)
            split: do we want to split computation across 2 GPUs? (if pred and target are on different GPUs)
                default: False (assumes they are on same device and big enough to fit on one GPU)
        """
        super().__init__()
        self.ndim = spatial_dims
        if self.ndim not in {1, 2, 3}:
            raise ValueError(f"Unsupported ndim: {self.ndim}-d, only 1-d, 2-d, and 3-d inputs are supported")
        self.reduction = reduction
        self.image_noise_sigma = image_noise_sigma
        self.unsigned = unsigned

        self.kernel_size = kernel_size
        if self.kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {self.kernel_size}")

        # _kernel = look_up_option(kernel_type, kernel_dict)
        _kernel = kernel_dict[kernel_type]
        self.kernel = _kernel(self.kernel_size)
        self.kernel.requires_grad = False
        self.kernel_nd, self.kernel_vol = self.get_kernel_vol()   # get nD kernel and its volume
        self.smooth_nr = float(smooth_nr)
        self.smooth_dr = float(smooth_dr)
        self.checkpointing = checkpointing

    def get_kernel_vol(self):
        vol = self.kernel
        for _ in range(self.ndim - 1):
            vol = torch.matmul(vol.unsqueeze(-1), self.kernel.unsqueeze(0))
        return vol, torch.sum(vol)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            pred: the shape should be BNH[WD].
            target: the shape should be BNH[WD].
        Raises:
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].
        """
        if pred.ndim - 2 != self.ndim:
            raise ValueError(f"expecting pred with {self.ndim} spatial dimensions, got pred of shape {pred.shape}")
        if target.shape != pred.shape:
            raise ValueError(f"ground truth has differing shape ({target.shape}) from pred ({pred.shape})")
        
        if self.image_noise_sigma > 0:
            noise = (2*torch.rand_like(target)-1) * self.image_noise_sigma
            pred = pred + noise
            target = target + noise

        # sum over kernel
        def cc_checkpoint_fn(target, pred, kernel, kernel_vol):
            '''
            This function is used to compute the intermediate results of the loss.
            '''
            t2, p2, tp = target * target, pred * pred, target * pred
            kernel, kernel_vol = kernel.to(pred), kernel_vol.to(pred)
            # kernel_nd = self.kernel_nd.to(pred)
            kernels = [kernel] * self.ndim
            kernels_t = kernels_p = kernels
            kernel_vol_t = kernel_vol_p = kernel_vol
            # compute intermediates
            t_sum = separable_filtering(target, kernels=kernels_t)
            p_sum = separable_filtering(pred, kernels=kernels_p)
            t2_sum = separable_filtering(t2, kernels=kernels_t)
            p2_sum = separable_filtering(p2, kernels=kernels_p)
            tp_sum = separable_filtering(tp, kernels=kernels_t)  # use target device's output
            # average over kernel
            t_avg = t_sum / kernel_vol_t
            p_avg = p_sum / kernel_vol_p
            # normalized cross correlation between t and p
            # sum[(t - mean[t]) * (p - mean[p])] / std[t] / std[p]
            # denoted by num / denom
            # assume we sum over N values
            # num = sum[t * p - mean[t] * p - t * mean[p] + mean[t] * mean[p]]
            #     = sum[t*p] - sum[t] * sum[p] / N * 2 + sum[t] * sum[p] / N
            #     = sum[t*p] - sum[t] * sum[p] / N
            #     = sum[t*p] - sum[t] * mean[p] = cross
            # the following is actually squared ncc
            cross = (tp_sum.to(pred) - p_avg * t_sum.to(pred))  # on pred device
            t_var = torch.max(
                t2_sum - t_avg * t_sum, torch.as_tensor(self.smooth_dr, dtype=t2_sum.dtype, device=t2_sum.device)
            ).to(pred)
            p_var = torch.max(
                p2_sum - p_avg * p_sum, torch.as_tensor(self.smooth_dr, dtype=p2_sum.dtype, device=p2_sum.device)
            )
            # if unsigned, just use the squared cc
            # if signed, sqrt the var terms and use a single cross
            if self.unsigned:
                ncc: torch.Tensor = (cross * cross + self.smooth_nr) / ((t_var * p_var) + self.smooth_dr)
            else:
                ncc: torch.Tensor = (cross + self.smooth_nr) / (torch.sqrt(t_var) * torch.sqrt(p_var) + self.smooth_dr)
            return ncc
        
        if self.checkpointing:
            ncc = checkpoint(cc_checkpoint_fn, target, pred, self.kernel, self.kernel_vol)
        else:
            ncc = cc_checkpoint_fn(target, pred, self.kernel, self.kernel_vol)

        if mask is not None:
            maskmean = mask.flatten(2).mean(2)  # [B, N]
            for _ in range(self.ndim):
                maskmean = maskmean.unsqueeze(-1)  # [B, N, 1, 1, ...]
            ncc = ncc * mask / maskmean

        # clamp it
        ncc = torch.clamp(ncc, -1.0, 1.0)

        if self.reduction == 'sum':
            return torch.sum(ncc).neg()  # sum over the batch, channel and spatial ndims
        if self.reduction == 'none':
            return ncc.neg()
        if self.reduction == 'mean':
            return torch.mean(ncc).neg()  # average over the batch, channel and spatial ndims
        raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')


class NCC_vxm(nn.Module):
    """
    Local (over window) normalized cross correlation loss.
    """
    def __init__(self, win=None, ndims=3, inp_channels=1):
        # super(NCC_vxm, self).__init__()
        super().__init__()
        self.ndims = ndims
        win = [9] * ndims if win is None else (win if isinstance(win, (list, tuple)) else [win] * ndims)
        self.win = win
        # compute filters
        self.sum_filt = torch.ones([inp_channels, 1, *win]).to("cuda")
        self.pad_no = math.floor(win[0] / 2)
        self.stride = tuple([1] * ndims)
        self.padding = tuple([self.pad_no] * ndims)
        self.conv_fn = partial(getattr(F, 'conv%dd' % ndims), groups=inp_channels)

    def forward(self, y_true, y_pred):
        Ii = y_true
        Ji = y_pred
        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        #ndims = len(list(Ii.size())) - 2
        #assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims
        stride = self.stride
        padding = self.padding
        sum_filt = self.sum_filt
        win = self.win
        conv_fn = self.conv_fn

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_I * J_sum
        I_var = torch.max(I2_sum - u_I * I_sum, torch.tensor(1e-5).to(I2_sum.device))
        J_var = torch.max(J2_sum - u_J * J_sum, torch.tensor(1e-5).to(J2_sum.device))
        cc = (cross * cross) / (I_var * J_var + 1e-5)

        # cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        # I_var = torch.clamp(I2_sum - u_I * u_I * win_size, 1e-5)
        # J_var = torch.clamp(J2_sum - u_J * u_J * win_size, 1e-5)

        # cc = torch.clamp((cross / I_var) * (cross / J_var), -1, 1)
        # print(cc.min(), cc.max())
        # I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        # J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size
        # cc = cross * cross / (I_var * J_var + 1e-5)
        return -torch.sum(cc)


class DiceLossWithLongLabels(nn.Module):
    ''' 
    Dice loss with long labels (to prevent explosive memory requirements for huge label maps) 
    This requires us to provide the warp field as well (to warp the labels and pass gradients through the warp)
    '''
    def __init__(self, min_label=1, max_label=35, eps=1e-6, order=2, intersection_only=False, center_dice_threshold=None, l2_mode=False):
        # super(DiceLossWithLongLabels, self).__init__()
        super().__init__()
        self.min_label = min_label
        self.max_label = max_label
        self.eps = eps
        self.order = order
        self.intersection_only = intersection_only
        print("Using Dice score with order {}, min_label {}, max_label {}".format(self.order, self.min_label, self.max_label))
        # to keep track of label centers
        self.center_dist = []                   # center distance of that region
        self.center_dice_scores = []            # actual dice score of that region
        self.center_dice_threshold = center_dice_threshold       # threshold to choose to compute center mse
        print("Center dice threshold: ", self.center_dice_threshold)
        self.l2_mode = l2_mode
    
    def get_center_mse(self):
        return self.center_dist, self.center_dice_scores
    
    def add_to_center_stat(self, coord, dice_score, y_fixed_i, y_moving_i_warped):
        if self.center_dice_threshold is not None and dice_score < self.center_dice_threshold:
            fixed_img_coord = (coord * y_fixed_i).flatten(2).sum(2) / (y_fixed_i.flatten(2).sum(2) + self.eps)  # [B, 3]
            moving_img_coord = (coord * y_moving_i_warped).flatten(2).sum(2) / (y_moving_i_warped.flatten(2).sum(2) + self.eps)  # [B, 3]
            self.center_dice_scores.append(dice_score)
            # print(dice_score, fixed_img_coord, moving_img_coord)
            self.center_dist.append(F.mse_loss(fixed_img_coord, moving_img_coord, reduction='mean'))
    
    def forward(self, y_moving, y_fixed, warp, train=True):
        # warp is assumed to be the actual warp field (not the displacement field)
        # y_moving/y_fixed are of size [B, 1, H, W, D]
        # warp is of size [B, H, W, D, 3]
        self.center_dice_scores = []
        self.center_dist = []

        coord = None
        if self.center_dice_threshold is not None:
            n_dims = len(y_moving.shape) - 2
            coord = F.affine_grid(torch.eye(n_dims, n_dims+1, device=y_fixed.device).unsqueeze(0), y_fixed.size())  # [1, H, W, D, 3] or [1, H, W, 2]
            # for i in range(n_dims):
            #     coord[..., i] = coord[..., i] * (y_fixed.size(-1 - i) - 1) / 2
            permute_order = [0] + [1+n_dims] + [i+1 for i in range(n_dims)]
            coord = coord.permute(*permute_order)  # [1, 3, H, W, D] or [1, 2, H, W, D]

        if self.intersection_only:
            dice_losses = []
            common_labels = set(torch.unique(y_moving.reshape(-1)).long().tolist()).intersection(set(torch.unique(y_fixed.reshape(-1)).long().tolist()))
            common_labels = list(sorted(list(common_labels)))
            # if train:
            #     self.last_labels = [i for i in common_labels if i >= self.min_label and i <= self.max_label]
            ### compute dice score
            for i in common_labels:
                if i < self.min_label or i > self.max_label:
                    continue
                y_moving_i = (y_moving == i).float()
                y_fixed_i = (y_fixed == i).float()
                y_moving_i_warped = F.grid_sample(y_moving_i, warp, align_corners=align_corners) if warp is not None else y_moving_i
                if not train:
                    y_moving_i_warped = (y_moving_i_warped >= 0.5).float()
                
                # compute dice loss if inference or if l2_mode is off
                if not train or not self.l2_mode:
                    # average over all pixels
                    intersection = (y_fixed_i * y_moving_i_warped).flatten(1).sum(1)   # [B, ]
                    if self.order == 1:
                        union = (y_fixed_i + y_moving_i_warped).flatten(1).sum(1)  # [B, ]
                    else:
                        union = (y_fixed_i**2 + y_moving_i_warped**2).flatten(1).sum(1)  # [B, ]
                    dice_score = ((2 * intersection + self.eps) / (union + self.eps)).mean()   # mean over batch
                    dice_losses.append(1 - dice_score)
                    # if center dice threshold is set, compute the center distance
                    self.add_to_center_stat(coord, dice_score.item(), y_fixed_i, y_moving_i_warped)
                else:
                    dice_losses.append(F.mse_loss(y_fixed_i, y_moving_i_warped, reduction='mean'))
                    # center stat not implemented for L2 mode
        else:
            dice_losses = []
            for i in range(self.min_label, self.max_label+1):
                y_moving_i = (y_moving == i).float()
                y_fixed_i = (y_fixed == i).float()
                y_moving_i_warped = F.grid_sample(y_moving_i, warp, align_corners=align_corners) if warp is not None else y_moving_i
                if not train:
                    y_moving_i_warped = (y_moving_i_warped >= 0.5).float()
                
                if not train or not self.l2_mode:
                    # average over all pixels
                    intersection = (y_fixed_i * y_moving_i_warped).flatten(1).sum(1)   # [B, ]
                    if self.order == 1:
                        union = (y_fixed_i + y_moving_i_warped).flatten(1).sum(1)
                    else:
                        union = (y_fixed_i**2 + y_moving_i_warped**2).flatten(1).sum(1)  # [B, ]
                    dice_score = ((2 * intersection + self.eps) / (union + self.eps)).mean()
                    dice_losses.append(1 - dice_score)
                    # add to center stats
                    self.add_to_center_stat(coord, dice_score.item(), y_fixed_i, y_moving_i_warped)
                else:
                    dice_losses.append(F.mse_loss(y_fixed_i, y_moving_i_warped, reduction='mean'))

        return dice_losses


def _get_loss_function_factory(loss_name, cfg, spatial_dims=3):
    ''' get appropriate loss function for the task '''
    if loss_name == 'l2':
        loss_fn = F.mse_loss
    elif loss_name == 'ncc_vxm':
        loss_fn = NCC_vxm(win=7, inp_channels=cfg.model.output_channels, ndims=spatial_dims)
    elif loss_name == 'ncc':
        loss_fn = LocalNormalizedCrossCorrelationLoss(
            spatial_dims=spatial_dims,
            kernel_size=7,
            reduction='sum',
        )
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")
    return loss_fn
