''' utilities for the model to run on, etc. '''
import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Optional
from solver.losses import ItemOrList, gaussian_1d, separable_filtering
import copy

def interpolate_warp(warp: torch.Tensor, size: List[int]):
    ''' function to interpolate a warp '''
    dims = len(warp.shape) - 2
    if dims == 2:
        B, H, W, _ = warp.shape
        warpimg = F.interpolate(warp.permute(0, 3, 1, 2), size=size, mode='bilinear', align_corners=True)
        return warpimg.permute(0, 2, 3, 1)
    elif dims == 3:
        B, H, W, D, _ = warp.shape
        warpimg = F.interpolate(warp.permute(0, 4, 1, 2, 3), size=size, mode='trilinear', align_corners=True)
        return warpimg.permute(0, 2, 3, 4, 1)
    else:
        raise NotImplementedError

def displacements_to_warps(displacements):
    ''' given a list of displacements, add warps to them '''
    warps = []
    for disp in displacements:
        # disp is of shape [batch, H, W, D, 3] or [batch, H, W, 2]
        shape = disp.shape[1:-1]
        dims = len(shape)
        grid = F.affine_grid(torch.eye(dims, dims+1, device=disp.device).unsqueeze(0), [1, 1] + list(shape), align_corners=True)
        warps.append(grid + disp)
    return warps

def downsample(image: ItemOrList[torch.Tensor], size: List[int], mode: str, sigma: Optional[torch.Tensor]=None,
               gaussians: Optional[torch.Tensor] = None) -> torch.Tensor:
    ''' 
    this function is to downsample the image to the given size
    but first, we need to perform smoothing 
    if sigma is provided (in voxels), then use this sigma for downsampling, otherwise infer sigma
    '''
    if gaussians is None:
        if sigma is None:
            orig_size = list(image.shape[2:])
            sigma = [0.5 * orig_size[i] / size[i] for i in range(len(orig_size))]   # use sigma as the downsampling factor
        sigma = torch.tensor(sigma, dtype=torch.float32, device=image.device)
        # create gaussian convs
        gaussians = [gaussian_1d(s, truncated=2) for s in sigma]
    # otherwise gaussians is given, just downsample
    image_smooth = separable_filtering(image, gaussians)
    image_down = F.interpolate(image_smooth, size=size, mode=mode, align_corners=True)
    return image_down


## Write an EMA class that copies a model and updates the weights
class EMA:
    def __init__(self, model: torch.nn.Module, decay: float, device: torch.device):
        self.model = copy.deepcopy(model)
        self.decay = decay
        self.device = device
        self.model.to(self.device)
        self.model.eval()

    def update(self, model: torch.nn.Module):
        with torch.no_grad():
            for ema_param, model_param in zip(self.model.parameters(), model.parameters()):
                ema_param.data.mul_(self.decay).add_(model_param.data, alpha=1 - self.decay)

    def to(self, device: torch.device):
        self.device = device
        self.model.to(device)
