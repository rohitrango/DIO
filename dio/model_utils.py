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

def scaling_and_squaring(u, grid=None, n = 6):
    """
    Apply scaling and squaring to a displacement field
    
    :param u: Input stationary velocity field, PyTorch tensor of shape [B, D, H, W, 3] or [B, H, W, 2]
    :param grid: Sampling grid of size [B, D, H, W, dims]  or [B, H, W, dims]
    :param n: Number of iterations of scaling and squaring (default: 6)
    
    :returns: Output displacement field, v, PyTorch tensor of shape [B, D, H, W, dims] or [B, H, W, dims]
    """
    dims = u.shape[-1]
    v = (1.0/2**n) * u
    if grid is None:
        grid = F.affine_grid(torch.eye(dims, dims+1, device=u.device).unsqueeze(0), [1, 1] + list(u.shape[1:-1]), align_corners=True)

    if dims == 3:
        for i in range(n):
            vimg = v.permute(0, 4, 1, 2, 3)          # [1, 3, D, H, W]
            v = v + F.grid_sample(vimg, v + grid, align_corners=True).permute(0, 2, 3, 4, 1)
    elif dims == 2:
        for i in range(n):
            vimg = v.permute(0, 3, 1, 2)
            v = v + F.grid_sample(vimg, v + grid, align_corners=True).permute(0, 2, 3, 1)
    else:
        raise ValueError('Invalid dimension: {}'.format(dims))
    return v

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
    def __init__(self, model: torch.nn.Module, decay: float, device: Optional[torch.device] = None):
        if device is None:
            device = next(model.parameters()).device
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
