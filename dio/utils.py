''' Copied from Rahul's starter template '''
import torch
import wandb
import numpy as np
import random
import os
import sys
import builtins
from torch import nn
from torch.nn import functional as F

from omegaconf import OmegaConf

def set_seed(seed=0):
    """
    Don't set true seed to be nearby values. Doesn't give best randomness
    """
    rng = np.random.default_rng(seed)
    true_seed = int(rng.integers(2**30))
    random.seed(true_seed)
    np.random.seed(true_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(true_seed)
    torch.cuda.manual_seed_all(true_seed)

def open_log(cfg, rank=0):
    os.makedirs('logs/' + cfg.tag, exist_ok=True)
    print(OmegaConf.to_yaml(cfg))
    orig_print = builtins.print
    if cfg.deploy:
        fname = 'logs/' + cfg.tag + '/' + wandb.run.name + "_{}".format(rank) + ".log"
        fout = open(fname, "a", 1)
        # sys.stdout = fout
        # sys.stderr = fout
        def new_print(*args, **kwargs):
            orig_print(*args, **kwargs)
            print(*args, **kwargs, file=fout)
        print(OmegaConf.to_yaml(cfg))
        return new_print, fout
    return orig_print, None

def init_wandb(cfg, project_name, rank=0):
    # rank is zero by default
    group = cfg.exp_name + "_group" if cfg.ddp.enabled else None
    cfg.save_dir = 'saved_models/' + cfg.tag + '/' + cfg.exp_name + '/'
    if cfg.deploy:
        wandb.init(project=project_name, tags=[cfg.tag], group=group)
        wandb.run.name = wandb.run.id if cfg.exp_name is None else cfg.exp_name
        cfg.save_dir = 'saved_models/' + cfg.tag + '/' + wandb.run.name + '/'
        # add saved path
        if rank == 0:
            wandb.run.save()
            cfg.save_dir = 'saved_models/' + cfg.tag + '/' + wandb.run.name + '/'
            os.makedirs(cfg.save_dir, exist_ok=True)
            wandb.config.update(OmegaConf.to_container(cfg))
        print(f"Running wandb with id: {wandb.run.id} and name: {wandb.run.name}.")


def cleanup(cfg, fp, rank=0):
    if cfg.deploy:
        fp.close()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        # if rank == 0:
        wandb.finish()

def resolve_layer_idx(epoch, cfg):
    ''' resolve the layer index to use for the current epoch '''
    epoch_new_level = cfg.train.train_new_level
    idx = 0
    for lvl in epoch_new_level:
        if epoch >= lvl:
            idx += 1
    return idx

def reshape_util(feature, channels):
    ''' feature: [B, C, H, W] '''
    B, C, H, W = feature.shape
    if C < channels:
        pass
    else:
        feature = feature.reshape(B, C//channels, channels, H, W).mean(1)  # [B, channels, H, W]
    return feature

def compute_detJac_2d(warp):
    # warp : [B, H, W, 2]
    B = warp.shape[0]
    H, W = warp.shape[1:-1]
    dx = torch.from_numpy(np.array([[-1, 0, 1]])).float()[None, None].cuda()  # [1, 1, 1, 3]
    dy = dx.transpose(2, 3).clone()  # [1, 1, 3, 1]
    # scale dx and dy
    dx = dx / (4/(H))
    dy = dy / (4/(W))
    jac = torch.zeros((B, H, W, 2, 2), device=warp.device)
    dxs = [dx, dy]
    paddings = [(0,1), (1,0)]
    for fi in range(2):
        for di in range(2):
            jac[..., fi, di] = F.conv2d(warp[:, None, ..., fi], dxs[di], padding=paddings[di])
    det = torch.linalg.det(jac)
    return det[..., 1:-1, 1:-1]

def compute_detJac_3d(warp):
    # warp: [B, H, W, D, 3]
    B = warp.shape[0]
    H, W, D = warp.shape[1:-1]
    dx = torch.from_numpy(np.array([[-1, 0, 1]])).float()[None, None, None].cuda()  # [1, 1, 1, 1, 3]
    dy = dx.transpose(3, 4).clone()  # [1, 1, 1, 1, 3]
    dz = dx.transpose(2, 4).clone()
    # scale them
    dx = dx / (4/(D))
    dy = dy / (4/(H))
    dz = dz / (4/(W))
    jac = torch.zeros((B, H, W, D, 3, 3), device=warp.device)
    dxs = [dx, dy, dz]
    paddings = [(0,0,1), (0,1,0), (1,0,0)]
    for fi in range(3):
        for di in range(3):
            jac[..., fi, di] = F.conv3d(warp[:, None, ..., fi], dxs[di], padding=paddings[di])
    det = torch.linalg.det(jac)
    return det[..., 1:-1, 1:-1, 1:-1]
    
def compute_detJac(warp):
    dims = warp.shape[-1]
    if dims == 2:
        return compute_detJac_2d(warp)
    elif dims == 3:
        return compute_detJac_3d(warp)
    else:
        raise NotImplementedError
