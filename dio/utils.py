''' Copied from Rahul's starter template '''
import torch
import wandb
import numpy as np
import random
import os
import sys
import builtins

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