import torch

def tv_loss_fn(warps):
    '''
    warps: list of [B, D, H, W, 3]
    '''
    loss = []
    for warp in warps:
        loss.append(tv_loss_fn_single(warp))
    return torch.stack(loss).mean()

def tv_loss_fn_single(warp):
    '''
    warp: [B, 3, D, H, W]
    '''
    # compute the gradient of the warp
    H, W, D = warp.shape[1:-1]
    gx, gy, gz = torch.gradient(warp, dim=[1, 2, 3], spacing=[1, 1, 1])
    gradmag = (H**2*gx**2 + W**2*gy**2 + D**2*gz**2)  # make the gradient magnitude invariant to the spacing
    gradmag /= 4
    return gradmag.mean()
