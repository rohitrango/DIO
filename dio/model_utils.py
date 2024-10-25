''' utilities for the model to run on, etc. '''
import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Optional
from solver.losses import ItemOrList, gaussian_1d, separable_filtering

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