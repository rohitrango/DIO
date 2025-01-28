''' 
List of affine solvers 

'''
import torch
from torch.nn import functional as F
from packaging import version

if version.parse(torch.__version__) <= version.parse('1.9.0'):
    from cuda_gridsample_grad2_py19.cuda_gridsample import grid_sample_3d, grid_sample_2d
    print("Loaded for py19 version")
else:
    print("Loaded for py19+ version")
    from cuda_gridsample_grad2.cuda_gridsample import grid_sample_3d, grid_sample_2d

class IFTGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, func, affine):
        ctx.func = func
        ctx.save_for_backward(affine.detach())
        return affine
    
    @staticmethod
    def backward(ctx, grad_affine):
        func = ctx.func
        affine_pred = ctx.saved_tensors[0].clone().detach().requires_grad_(True)
        with torch.enable_grad():
            affine = func(affine_pred)
        grad_f = lambda x: torch.autograd.grad(affine, affine_pred, x, retain_graph=True)[0] + grad_affine
        grad_f_opt = grad_affine
        for it in range(100):
            grad_f_opt_new = grad_f(grad_f_opt)
            if torch.isnan(grad_f_opt_new).any():
                break
            grad_f_opt = torch.clamp(grad_f_opt_new, -0.1, 0.1)
        return None, grad_f_opt
        
def run_affine_transform_2d(fixed_features, moving_features, iterations, init_affine=None, debug=True):
    ''' initialize affine transform and run optimization 
    fixed_features, moving_features: [B, C, H, W]
    iterations: int
    '''
    batch_size = fixed_features.shape[0]
    if init_affine is not None:
        affine_map = init_affine.clone().detach().requires_grad_(True)
    else:
        affine_map = torch.eye(2, 3)[None].cuda().repeat(batch_size, 1, 1)  # [B, 2, 3]
        affine_map = affine_map.requires_grad_(True)
    losses_opt = []
    # define function
    def f(affine_map, lr=0.1, log=False):
        ''' affine_map: [B, 2, 3] '''
        with torch.set_grad_enabled(True):
            grid = F.affine_grid(affine_map, fixed_features.shape, align_corners=True)
            # moved_features = F.grid_sample(moving_features, grid, align_corners=True)
            moved_features = grid_sample_2d(moving_features, grid, align_corners=True)
            loss = F.mse_loss(moved_features, fixed_features)
            if log:
                losses_opt.append(loss.item())
            affine_grad = torch.autograd.grad(loss, affine_map, create_graph=True)[0]
            # SGD update
            affine_map = affine_map - lr * affine_grad
        return affine_map
    
    # forward pass
    with torch.no_grad():
        for it in range(iterations):
            affine_map = f(affine_map, log=True)
    # attach ift hook
    affine_map = affine_map.clone().detach().requires_grad_(True)
    for i in range(3):
        affine_map = f(affine_map, log=False)
    # affine_map = IFTGrad.apply(f, affine_map)
    return affine_map, losses_opt

def run_affine_transform_3d(fixed_features, moving_features, iterations, init_affine=None, debug=True, lr=0.1, phantom_steps=1):
    ''' initialize affine transform and run optimization 
    fixed_features, moving_features: [B, C, H, W, D]
    iterations: int
    '''
    batch_size = fixed_features.shape[0]
    if init_affine is not None:
        affine_map = init_affine
    else:
        affine_map = torch.eye(3, 4)[None].cuda().repeat(batch_size, 1, 1)  # [B, 2, 3]
        affine_map = affine_map.requires_grad_(True)
    losses_opt = []
    # define function
    def f(affine_map, log=False):
        ''' affine_map: [B, 2, 3] '''
        with torch.set_grad_enabled(True):
            grid = F.affine_grid(affine_map, fixed_features.shape, align_corners=True)
            # moved_features = F.grid_sample(moving_features, grid, align_corners=True)
            moved_features = grid_sample_3d(moving_features, grid, align_corners=True)
            loss = F.mse_loss(moved_features, fixed_features)
            if log:
                losses_opt.append(loss.item())
            affine_grad = torch.autograd.grad(loss, affine_map, create_graph=True)[0]
            # SGD update
            affine_map = affine_map - lr * affine_grad
        return affine_map
    
    # forward pass
    with torch.no_grad():
        for it in range(iterations):
            affine_map = f(affine_map, log=True)
    # attach ift hook
    # affine_map = affine_map.clone().detach().requires_grad_(True)
    for i in range(phantom_steps):
        affine_map = f(affine_map, log=False)
    return affine_map, losses_opt

def multi_scale_affine2d_solver(fixed_features, moving_features, iterations, loss_function, init_affine=None, debug=True, **kwargs):
    init_affine = None
    warps = []
    losses = []
    num_levels = list(range(len(fixed_features)))
    for i in num_levels:
        fixed = fixed_features[i]
        moving = moving_features[i]
        iters = iterations[i]
        affine, loss = single_scale_affine2d_solver(fixed, moving, iters, loss_function, init_affine, debug, **kwargs)
        warp = F.affine_grid(affine, fixed.shape, align_corners=True)
        warps.append(warp)
        losses.append(loss)
        init_affine = affine
    if debug:
        return warps, losses
    else:
        return warps

def single_scale_affine2d_solver(fixed_features, moving_features, iterations, loss_function, init_affine=None, debug=True, **kwargs):
    ''' initialize affine transform and run optimization 
    fixed_features, moving_features: [B, C, H, W]
    iterations: int
    '''
    # actually single scale here
    # fixed_features, moving_features = fixed_features[0], moving_features[0]
    #  get batch size and initialize affine
    batch_size = fixed_features.shape[0]
    if init_affine is not None:
        affine_map = init_affine.clone().detach().requires_grad_(True)
    else:
        affine_map = torch.eye(2, 3)[None].cuda().repeat(batch_size, 1, 1)  # [B, 2, 3]
        affine_map = affine_map.requires_grad_(True)
    losses_opt = []
    # define function
    def f(affine_map, lr=0.1, log=False):
        ''' affine_map: [B, 2, 3] '''
        with torch.set_grad_enabled(True):
            grid = F.affine_grid(affine_map, fixed_features.shape, align_corners=True)
            # moved_features = F.grid_sample(moving_features, grid, align_corners=align_corners)
            moved_features = grid_sample_2d(moving_features, grid, align_corners=True)
            loss = loss_function(moved_features, fixed_features)
            if log:
                losses_opt.append(loss.item())
            affine_grad = torch.autograd.grad(loss, affine_map, create_graph=True)[0]
            # SGD update
            affine_map = affine_map - lr * affine_grad
        return affine_map
    
    # forward pass
    with torch.no_grad():
        for it in range(iterations):
            affine_map = f(affine_map, log=True)
    # attach ift hook
    # affine_map = affine_map.clone().detach().requires_grad_(True)
    for i in range(3):
        affine_map = f(affine_map, log=False)
    # affine_map = IFTGrad.apply(f, affine_map)
    ### get affine map to displacement map
    # displacement_map = F.affine_grid(affine_map, fixed_features.shape, align_corners=align_corners)
    # return displacement_map, losses_opt
    return affine_map, losses_opt