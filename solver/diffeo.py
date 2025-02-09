'''
author: rohitrango

Implements multi-scale diffeomorphic riemannian adam / sgd
'''
import math
import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Tuple, Optional, Union, Callable
# from solver.losses import _get_loss_function_factory
from solver.utils import v2img_2d, v2img_3d, img2v_2d, img2v_3d, separable_filtering, ItemOrList
from solver.affine import run_affine_transform_3d
from packaging import version
import logging
from logging import getLogger
logging.basicConfig(level=logging.INFO)

logger = getLogger(__name__)
logger.info("Using torch version %s", torch.__version__)

if version.parse(torch.__version__) <= version.parse('1.9.0'):
    print("Loading py19")
    from cuda_gridsample_grad2_py19.cuda_gridsample import grid_sample_3d, grid_sample_2d
    print("Loaded for py19 version")
else:
    print("Loading py19+")
    from cuda_gridsample_grad2.cuda_gridsample import grid_sample_3d, grid_sample_2d

import numpy as np

ALIGN_CORNERS = False
align_corners = ALIGN_CORNERS

class MultTensorLayer(torch.autograd.Function):
    ''' Multiply tensor A with arbitrary factor B to return C = (A * B) 
    dA = dC * B  is the backprop rule, but we will simply take dA = dC  (to avoid scaling artifacts from B)
    '''
    @staticmethod
    def forward(ctx, tensor, multiply_tensor):
        result = tensor * multiply_tensor
        return result

    @staticmethod
    def backward(ctx, grad_result):
        return grad_result, None

no_backprop_mult = MultTensorLayer.apply

def multi_scale_diffeomorphic_solver(
        fixed_features: List[torch.Tensor],
        moving_features: List[torch.Tensor],
        iterations: List[int],
        loss_function: Union[nn.Module, Callable],
        hessian_type: str = 'jfb',
        gaussian_warp: Optional[ItemOrList[torch.Tensor]] = None,
        gaussian_grad: Optional[ItemOrList[torch.Tensor]] = None,
        learning_rate: float = 1,
        debug: bool = True,
        displacement_loss_fn: Optional[Callable] = None,
        # beta1: float = 0.5,
        beta1: float = 0.9,
        beta2: float = 0.99,
        eps: float = 1e-8,
        n_phantom_steps: int = 3,
        phantom_step: str = 'adam',   # choices = sgd, adam
        return_jacobian_norm: int = 1,
        convergence_tol: int = 4,       # if loss increases for "C" iterations, abort
        convergence_eps: float = 1e-3,
        cfg: Optional[dict] = None
):
    '''
    Implements multi-scale diffeomorphic riemannian adam for feature images

    `fixed_features` contain  images of increasing resolutions of size [B, C_i, H_i, W_i, [D_i]] where i is the scale,
        and C_i is the number of channels at that scale
    '''
    hessian_type = hessian_type.lower()
    ## this job is delegated to the training loop
    # if isinstance(loss_function, str):
    #     loss_function = _get_loss_function_factory(loss_function, cfg)
    # collect statistics
    batch_size, shape = fixed_features[0].shape[0], fixed_features[0].shape[2:]
    n_dims = len(shape)
    # initialize flow
    warp = torch.zeros((batch_size, *shape, n_dims), dtype=torch.float32, device=fixed_features[0].device)
    exp_avg = torch.zeros_like(warp)
    exp_sq_avg = torch.zeros_like(warp)
    all_warps = []
    global_step = 1

    # set functions for v2img and img2v
    v2img = v2img_2d if n_dims == 2 else v2img_3d
    img2v = img2v_2d if n_dims == 2 else img2v_3d
    grid_sample_fn = grid_sample_2d if n_dims == 2 else grid_sample_3d

    losses = []

    # iterate over scales
    # level is the level of iteration in the pyramid
    for level, (iter_scale, (fixed_feature, moving_feature)) in enumerate(zip(iterations, zip(fixed_features, moving_features))):
        losses_lvl = []
        # run optimization for this scale
        # half_res = 1.0/(max(fixed_feature.shape[2:]))/np.sqrt(3)
        half_res = 1.0/(max(fixed_feature.shape[2:])-1)
        grid = F.affine_grid(torch.eye(n_dims, n_dims+1, device=fixed_feature.device).unsqueeze(0).expand(batch_size, -1, -1), fixed_feature.shape, align_corners=align_corners)
        # run optimization without grad
        warp.requires_grad_(True)
        exp_avg = exp_avg.detach()
        exp_sq_avg = exp_sq_avg.detach()
        last_loss = np.inf
        iters_since_divergent = 0
        # run optimization
        with torch.no_grad():
            for step in range(1, iter_scale+1):
                # temporarily enable gradient here
                with torch.enable_grad():
                    moved_feature = F.grid_sample(moving_feature.detach(), grid + warp, align_corners=align_corners)
                    loss = loss_function(moved_feature, fixed_feature.detach())
                    if debug:
                        losses_lvl.append(loss.item())
                    
                    if displacement_loss_fn is not None:
                        loss_disp = displacement_loss_fn(warp)
                        loss = loss + loss_disp

                    warp_grad = torch.autograd.grad(loss, warp)[0].detach()
                # divergence check
                lossitem = loss.item()
                # if lossitem > last_loss:
                rel_loss = lossitem/(1e-8 + last_loss) - 1
                if rel_loss < -convergence_eps:
                    ## (loss - loss_prev)/loss_prev should be negative, and should decrease by at least -eps each time
                    iters_since_divergent = 0
                else:
                    iters_since_divergent += 1
                    if iters_since_divergent >= convergence_tol:
                        break
                last_loss = lossitem 
                # augment
                if gaussian_grad is not None:
                    warp_grad = img2v(separable_filtering(v2img(warp_grad), gaussian_grad))
                
                if phantom_step == 'adam':
                    # now that we have warp grad, update exp_avg and exp_sq_avg
                    exp_avg.mul_(beta1).add_(warp_grad, alpha=1-beta1)
                    exp_sq_avg.mul_(beta2).addcmul_(warp_grad, warp_grad.conj(), value=1-beta2)
                    b1_correction = 1 - beta1 ** global_step
                    b2_correction = 1 - beta2 ** global_step
                    denom = (exp_sq_avg / b2_correction).sqrt().add_(eps)
                    # get updated gradient for adam
                    warp_grad = exp_avg / b1_correction / denom
                # normalize
                gradmax = eps + warp_grad.norm(p=2, dim=-1, keepdim=True).flatten(1).max(1).values
                gradmax = gradmax.reshape(-1, *([1])*(n_dims+1))
                gradmax = gradmax.clamp(min=1)
                warp_grad = warp_grad / gradmax * half_res
                warp_grad.mul_(-learning_rate)
                # update function
                warp_update = warp_grad + img2v(F.grid_sample(v2img(warp), grid + warp_grad, align_corners=align_corners))
                ## NOTE: This step is done in the beginning
                ## optionally smooth it
                if gaussian_warp is not None:
                    warp_update = img2v(separable_filtering(v2img(warp_update), gaussian_warp))
                warp.data.copy_(warp_update)
                global_step += 1

        # final step to capture gradient (here, warp = warp*)
        # warp.requires_grad_(Fals)
        jacobian_norm = torch.tensor(0).to(warp.device).float()
        if hessian_type == 'jfb':
            ### JFB: Jacobian-free backprop - essentially pretend to perform one-step optimization
            # simply perform another forward pass (with torch enabled grad)
            # moved_feature = F.grid_sample(moving_feature, grid + warp, align_corners=align_corners)
            for step in range(n_phantom_steps):
                moved_feature = grid_sample_fn(moving_feature, grid + warp, align_corners=align_corners)
                loss = loss_function(moved_feature, fixed_feature)
                if debug:
                    losses_lvl.append(loss.item())
                warp_grad = torch.autograd.grad(loss, warp, create_graph=True)[0]
                if gaussian_grad is not None:
                    warp_grad = img2v(separable_filtering(v2img(warp_grad), gaussian_grad))
                # keep an old warp to compute jacobian
                warp_old = warp
                # we will NOT update exp_avg and exp_sq_avg here (we use the gradient directly as the update, and the sq_avg as the Hessian approximation)

                ## Algo 1: use no_backprop_mult
                # b1_correction = 1 - beta1 ** global_step
                # b2_correction = 1 - beta2 ** global_step
                # denom = (exp_sq_avg.sqrt() / math.sqrt(b2_correction)).add_(eps)
                # warp_grad = no_backprop_mult(warp_grad, b1_correction / denom)
                # # normalize
                # gradmax = eps + (warp_grad.detach()).norm(p=2, dim=-1, keepdim=True).flatten(1).max(1).values
                # gradmax = gradmax.reshape(-1, *([1])*(n_dims+1))
                # gradmax = gradmax.clamp(min=1)
                # ### multiply the learning_rate times gradmax but dont include it in gradient
                # warp_grad = no_backprop_mult(warp_grad, learning_rate / gradmax * half_res)
                # warp_grad = -warp_grad

                ## Algo 2: SGD phantom
                if phantom_step == 'sgd':
                    # gradmax = eps + warp_grad.detach().norm(p=2, dim=-1, keepdim=True).flatten(1).max(1).values
                    gradmax = eps + warp_grad.norm(p=2, dim=-1, keepdim=True).flatten(1).max(1).values
                    gradmax = gradmax.reshape(-1, *([1])*(n_dims+1))
                    gradmax = gradmax.clamp(min=1)
                    warp_grad = -warp_grad / gradmax * half_res * learning_rate

                ## Algo 3: Adam phantom
                elif phantom_step == 'adam':
                    exp_avg = beta1 * exp_avg + (1 - beta1) * warp_grad
                    # warp_grad = warp_grad / b1_correction / denom 
                    warp_grad = exp_avg / b1_correction / denom 
                    gradmax = eps + warp_grad.norm(p=2, dim=-1, keepdim=True).flatten(1).max(1).values
                    gradmax = gradmax.reshape(-1, *([1])*(n_dims+1))
                    gradmax = gradmax.clamp(min=1)
                    warp_grad = -warp_grad / gradmax * half_res * learning_rate

                # update function
                warp = warp_grad + img2v(grid_sample_fn(v2img(warp), grid + warp_grad, align_corners=align_corners))
                if gaussian_warp is not None:
                    warp = img2v(separable_filtering(v2img(warp), gaussian_warp))
                # compute jacobian norm
                for _ in range(return_jacobian_norm):
                    v = torch.randn_like(warp)
                    vJ = torch.autograd.grad(warp, warp_old, v, create_graph=True, retain_graph=True)[0]
                    jacobian_norm = jacobian_norm + (vJ.norm()**2).mean() / np.prod(v.shape)
            
        elif hessian_type == 'adam':
            raise NotImplementedError('Adam hessian not implemented yet')
        else:
            raise ValueError(f'Unknown hessian type {hessian_type}')

        # add this to all_warps 
        all_warps.append(warp)
        losses.append(losses_lvl)
        # interpolate for next stage
        if level != len(iterations) - 1:
            new_shape = fixed_features[level+1].shape[2:]
            warp = img2v(F.interpolate(v2img(warp.detach()), size=new_shape, mode='bilinear' if n_dims == 2 else 'trilinear', align_corners=align_corners))
            exp_avg = img2v(F.interpolate(v2img(exp_avg), size=new_shape, mode='bilinear' if n_dims == 2 else 'trilinear', align_corners=align_corners))
            exp_sq_avg = img2v(F.interpolate(v2img(exp_sq_avg), size=new_shape, mode='bilinear' if n_dims == 2 else 'trilinear', align_corners=align_corners))
    # return all_warps
    if debug:
        return all_warps, losses, jacobian_norm
    else:
        return all_warps, jacobian_norm

def multi_scale_warp_solver(
        fixed_features: List[torch.Tensor],
        moving_features: List[torch.Tensor],
        iterations: List[int],
        loss_function: Union[nn.Module, Callable],
        hessian_type: str = 'jfb',
        gaussian_warp: Optional[ItemOrList[torch.Tensor]] = None,
        gaussian_grad: Optional[ItemOrList[torch.Tensor]] = None,
        learning_rate: float = 3e-3,
        regularization: Optional[Callable] = None,
        debug: bool = False,
        beta1: float = 0.5,   # changing this from 0.9 to 0.5 to increase EMA decay for phantom steps to mimic actual optimization
        beta2: float = 0.99,
        eps: float = 1e-8,
        n_phantom_steps: int = 3,
        return_jacobian_norm: int = 1,  # how many estimators to compute
        phantom_step: str = 'sgd',   # choices = sgd, adam
        convergence_tol: int = 4,       # if loss increases for "C" iterations, abort
        convergence_eps: float = 1e-3,
        cfg: Optional[dict] = None,
        init_affine: Optional[torch.Tensor] = None,
):
    '''
    Implements multi-scale SGD for warp fields with arbitrary feature images
    `fixed_features` contain  images of increasing resolutions of size [B, C_i, H_i, W_i, [D_i]] where i is the scale,
        and C_i is the number of channels at that scale
    '''
    hessian_type = hessian_type.lower()
    # collect statistics
    batch_size, shape = fixed_features[0].shape[0], fixed_features[0].shape[2:]
    n_dims = len(shape)
    # initialize flow
    warp = torch.zeros((batch_size, *shape, n_dims), dtype=torch.float32, device=fixed_features[0].device)
    exp_avg = torch.zeros_like(warp)
    exp_sq_avg = torch.zeros_like(warp)
    all_warps = []
    global_step = 1
    # set functions for v2img and img2v
    v2img = v2img_2d if n_dims == 2 else v2img_3d
    img2v = img2v_2d if n_dims == 2 else img2v_3d
    grid_sample_fn = grid_sample_2d if n_dims == 2 else grid_sample_3d
    losses = []
    # iterate over scales
    # level is the level of iteration in the pyramid, i.e. max_levels = len(fixed_features) - 1

    for level, (iter_scale, (fixed_feature, moving_feature)) in enumerate(zip(iterations, zip(fixed_features, moving_features))):
        losses_lvl = []
        # initialize affine transform
        # this will typically have a gradient w.r.t. the affine parameters
        if init_affine is not None:
            pass
        else:
            init_affine = torch.eye(n_dims, n_dims+1, device=fixed_feature.device).unsqueeze(0).repeat(batch_size, 1, 1)
        # initialize grid
        grid = F.affine_grid(init_affine, fixed_feature.shape, align_corners=align_corners)
        # run optimization without grad
        warp.requires_grad_(True)
        exp_avg = exp_avg.detach()
        exp_sq_avg = exp_sq_avg.detach()
        # keep these variables to check for divergence and early-stop
        last_loss = np.inf
        iters_since_divergent = 0
        # run optimization
        with torch.no_grad():
            for step in range(1, iter_scale+1):
                # temporarily enable gradient here
                with torch.enable_grad():
                    moved_feature = F.grid_sample(moving_feature.detach(), grid + warp, align_corners=align_corners)
                    loss = loss_function(moved_feature, fixed_feature.detach())
                    if regularization is not None:
                        loss = loss + regularization(warp)
                    if debug:
                        losses_lvl.append(loss.item())
                    warp_grad = torch.autograd.grad(loss, warp)[0].detach()

                # divergence check
                lossitem = loss.item()
                # if lossitem > last_loss:
                rel_loss = lossitem/np.maximum(last_loss, 1e-8) - 1
                if rel_loss <= -convergence_eps:
                    ## (loss - loss_prev)/loss_prev should be negative, and should decrease by at least -eps each time
                    iters_since_divergent = 0
                else:
                    iters_since_divergent += 1
                    if iters_since_divergent >= convergence_tol:
                        break
                last_loss = lossitem
                # filtering
                if gaussian_grad is not None:
                    warp_grad = img2v(separable_filtering(v2img(warp_grad), gaussian_grad))
                # update SGD
                # now that we have warp grad, update exp_avg and exp_sq_avg
                if phantom_step == 'adam':
                    exp_avg.mul_(beta1).add_(warp_grad, alpha=1-beta1)
                    exp_sq_avg.mul_(beta2).addcmul_(warp_grad, warp_grad.conj(), value=1-beta2)
                    b1_correction = 1 - beta1 ** global_step
                    b2_correction = 1 - beta2 ** global_step
                    denom = (exp_sq_avg.sqrt() / math.sqrt(b2_correction)).add_(eps)
                    # # get updated gradient
                    warp_grad = exp_avg / b1_correction / denom
                    # # normalize
                    # gradmax = eps + warp_grad.norm(p=2, dim=-1, keepdim=True).flatten(1).max(1).values
                    # gradmax = gradmax.reshape(-1, *([1])*(n_dims+1))
                    # warp_grad = warp_grad / gradmax * half_res
                    # warp_grad.mul_(-learning_rate)
                    # update function
                    # warp_update = warp_grad + img2v(F.grid_sample(v2img(warp), grid + warp_grad, align_corners=align_corners))

                # If SGD, then we dont need all the postprocessing of the warp gradient
                warp_update = warp - learning_rate * warp_grad 
                # optionally smooth it
                if gaussian_warp is not None:
                    warp_update = img2v(separable_filtering(v2img(warp_update), gaussian_warp))
                warp.data.copy_(warp_update)
                global_step += 1

        # final step to capture gradient (here, warp = warp*)
        # warp.requires_grad_(False)
        jacobian_norm = torch.tensor(0).to(warp.device).float()
        if hessian_type == 'jfb':
            ### JFB: Jacobian-free backprop - essentially pretend to perform one-step optimization
            # simply perform another forward pass (with torch enabled grad)
            # moved_feature = F.grid_sample(moving_feature, grid + warp, align_corners=align_corners)
            for _ in range(n_phantom_steps):
                moved_feature = grid_sample_fn(moving_feature, grid + warp, align_corners=align_corners) 
                loss = loss_function(moved_feature, fixed_feature)
                if regularization is not None:
                    loss = loss + regularization(warp)
                if debug:
                    losses_lvl.append(loss.item())
                warp_grad = torch.autograd.grad(loss, warp, create_graph=True)[0]
                if gaussian_grad is not None:
                    warp_grad = img2v(separable_filtering(v2img(warp_grad), gaussian_grad))
                # we will NOT update exp_avg and exp_sq_avg here (we use the gradient directly as the update, and the sq_avg as the Hessian approximation)
                # now that we have warp grad, update exp_avg and exp_sq_avg

                ## Save this for jacobian norm
                warp_old = warp

                ### Algo 1: substitute exp_avg with warp_grad (doesnt work because the norm of warp_grad does not change)
                # warp_grad = no_backprop_mult(warp_grad, learning_rate / b1_correction / denom)

                ### Algo 2: find out the norm of updates of the warp, and rescale to that norm
                if phantom_step == 'sgd':
                    # oldnorm = (exp_avg / b1_correction / denom).norm() * learning_rate
                    # newnorm = warp_grad.norm()
                    # # add an extra term
                    # scale = (oldnorm / newnorm).item() # * min(1, (2**(max_levels - level)/4))
                    # # create new warp
                    # warp = warp - scale * warp_grad
                    warp = warp - learning_rate * warp_grad

                ### Algo 3: Perform the same update as in the iterations
                elif phantom_step == 'adam':
                    ### U2: I tried this as the update step
                    # exp_avg = beta1 * exp_avg + (1 - beta1) * warp_grad
                    # exp_sq_avg = beta2 * exp_sq_avg + (1 - beta2) * warp_grad * warp_grad
                    # denom = (exp_sq_avg / (1 - beta2 ** global_step)).sqrt() + eps
                    # warp_grad = exp_avg / (1 - beta1 ** global_step) / denom
                    # warp = warp - learning_rate * warp_grad

                    ### U1: Previously, I only matched the update equation without passing through the exp_avg and exp_sq_avg
                    ### U3: This works better!
                    warp_grad = warp_grad / b1_correction / denom * learning_rate
                    warp = warp - warp_grad
                else:
                    raise ValueError(f'Unknown phantom step {phantom_step}')

                if gaussian_warp is not None:
                    warp = img2v(separable_filtering(v2img(warp), gaussian_warp))
                # add to jac norm
                for _ in range(return_jacobian_norm):
                    v = torch.randn_like(warp)
                    vJ = torch.autograd.grad(warp, warp_old, v, create_graph=True, retain_graph=True)[0]
                    jacobian_norm = jacobian_norm + (vJ.norm()**2).mean() / np.prod(v.shape) / return_jacobian_norm
            
        elif hessian_type == 'adam':
            raise NotImplementedError('Adam hessian not implemented yet')
        else:
            raise ValueError(f'Unknown hessian type {hessian_type}')

        # add this to all_warps 
        all_warps.append(warp)
        losses.append(losses_lvl)
        # interpolate for next stage
        if level != len(iterations) - 1:
            new_shape = fixed_features[level+1].shape[2:]
            warp = img2v(F.interpolate(v2img(warp.detach()), size=new_shape, mode='bilinear' if n_dims == 2 else 'trilinear', align_corners=align_corners))
            exp_avg = img2v(F.interpolate(v2img(exp_avg.detach()), size=new_shape, mode='bilinear' if n_dims == 2 else 'trilinear', align_corners=align_corners))
            exp_sq_avg = img2v(F.interpolate(v2img(exp_sq_avg.detach()), size=new_shape, mode='bilinear' if n_dims == 2 else 'trilinear', align_corners=align_corners))
    # return all_warps
    # print([len(x) for x in losses])
    if debug:
        return all_warps, losses, jacobian_norm
    else:
        return all_warps, jacobian_norm


def multi_scale_affine3d_and_freeform_solver(
        fixed_features: List[torch.Tensor],
        moving_features: List[torch.Tensor],
        iterations: List[int],
        loss_function: Union[nn.Module, Callable],
        gaussian_warp: Optional[ItemOrList[torch.Tensor]] = None,
        gaussian_grad: Optional[ItemOrList[torch.Tensor]] = None,
        learning_rate: float = 3e-3,
        debug: bool = False,
        phantom_step: str = 'sgd',
        n_phantom_steps: int = 3,
        convergence_eps: float = 1e-3,
        hessian_type: str = 'jfb',
        cfg: Optional[dict] = None
):
    ''' we have routines for multi-scale affine3d and freeform solvers, just stitch them '''
    logger.warn("Using multi-scale affine3d and freeform solver")
    affine_map = None
    for level, (iter_scale, (fixed_feature, moving_feature)) in enumerate(zip(iterations, zip(fixed_features, moving_features))):
        affine_map, _ = run_affine_transform_3d(fixed_feature, moving_feature, iter_scale, init_affine=affine_map, lr=learning_rate)
        print(affine_map)

    # we have completed the affine transform (which stitches back gradients for multiscale), now run freeform
    logger.warn("Running freeform solver")
    ret = multi_scale_warp_solver(fixed_features, moving_features, iterations=iterations,
                                   loss_function=loss_function, hessian_type=hessian_type, gaussian_warp=gaussian_warp, gaussian_grad=gaussian_grad, learning_rate=learning_rate, debug=debug, 
                                    phantom_step=phantom_step, n_phantom_steps=n_phantom_steps, convergence_eps=convergence_eps, cfg=cfg, init_affine=affine_map)
    return ret

def multi_scale_affine3d_solver(
        fixed_features: List[torch.Tensor],
        moving_features: List[torch.Tensor],
        iterations: List[int],
        loss_function: Union[nn.Module, Callable],
        gaussian_warp: Optional[ItemOrList[torch.Tensor]] = None,
        gaussian_grad: Optional[ItemOrList[torch.Tensor]] = None,
        learning_rate: float = 3e-3,
        debug: bool = False,
        phantom_step: str = 'sgd',
        n_phantom_steps: int = 3,
        convergence_eps: float = 1e-3,
        hessian_type: str = 'jfb',
        cfg: Optional[dict] = None
):
    ''' we have routines for multi-scale affine3d and freeform solvers, just stitch them '''
    logger.warn("Using multi-scale affine3d and freeform solver")
    affine_map = None
    displacements = []
    losses_opt = []
    for level, (iter_scale, (fixed_feature, moving_feature)) in enumerate(zip(iterations, zip(fixed_features, moving_features))):
        affine_map, losses_opt_lvl = run_affine_transform_3d(fixed_feature, moving_feature, iter_scale, init_affine=affine_map, lr=learning_rate)
        print(affine_map)
        displacements.append(F.affine_grid(affine_map, fixed_feature.shape, align_corners=True))
        losses_opt.append(losses_opt_lvl)

    return displacements, losses_opt, 0