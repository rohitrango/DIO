''' Copied from `train_single_level.py` but meant for 2D images for better visualization and debugging '''
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 
# make sure the parent of this folder is in path to be 
# able to access everything
from TransMorph.models.TransMorph import TransFeX
from TransMorph.models.unet3d import UNet2D, UNet3D
from TransMorph.models.configs_TransMorph import get_3DTransFeX_config
from solver.adam import multi_scale_warp_solver, multi_scale_diffeomorphic_solver, multi_scale_affine2d_solver
from solver.utils import gaussian_1d, img2v_3d, v2img_3d, separable_filtering
from solver.losses import NCC_vxm, DiceLossWithLongLabels, _get_loss_function_factory
from solver.losses import LocalNormalizedCrossCorrelationLoss
from omegaconf import OmegaConf
from tqdm import tqdm
import nibabel as nib
from os import path as osp
from scipy.ndimage import gaussian_filter, zoom
from TransMorph.models.unet3d import UNet2D, UNet3D, UNetEncoder3D
from TransMorph.models.lku import LKUNet, LKUEncoder
import pickle as pkl

# logging
import wandb
import hydra
from model_utils import displacements_to_warps, downsample
from utils import set_seed, init_wandb, open_log, cleanup
from datasets.oasis import OASIS, OASISNeurite3D
import numpy as np
from kleindataloader import KleinDatasets

# datasets = {
#     'oasis': OASIS,
#     'neurite-oasis': OASISNeurite3D
# }

from train_multi_level_3d import resolve_layer_idx, try_retaingrad, torch2wandbimg, count_parameters

@hydra.main(config_path='./configs', config_name='default')
def mainfunc(cfg):
    # for dataset_name in ['IBSR18']:
        # for isotropic in [True, False]:
        #     for crop in [True, False]:
    for dataset_name in ['IBSR18', 'CUMC12', 'MGH10', 'LPBA40']:
        for isotropic in [True, False]:
            for crop in [True, False]:
                main(cfg, dataset_name, isotropic, crop)

def main(cfg, dataset_name, isotropic, crop):
    # init setup
    cfg.deploy = False
    # init_wandb(cfg, project_name='TransFeX')
    set_seed(cfg.seed)
    # fp = open_log(cfg)

    try:
        dry_run = cfg.dry_run
    except:
        dry_run = False

    print("Running dataset: {} with isotropic: {} and crop: {}".format(dataset_name, isotropic, crop))
    # load dataset
    train_dataset = KleinDatasets(dataset=dataset_name, isotropic=isotropic, crop=crop, dry_run=dry_run)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=1)
    print(f"Dataset has {len(train_dataset)} samples.")

    try:
        if cfg.model.combine:
            cfg.model.output_channels = cfg.model.output_channels * 2
            input_channels = 2
        else:
            input_channels = 1
    except:
        input_channels = 1

    if cfg.model.name == 'unet':
        model = UNet3D(input_channels, cfg.model.output_channels, f_maps=cfg.model.f_maps, levels=cfg.model.levels, skip=cfg.model.skip).cuda()
    elif cfg.model.name == 'transmorph':
        model_cfg = get_3DTransFeX_config()
        model_cfg['levels'] = list(cfg.model.levels)
        model_cfg['output_channels'] = cfg.model.output_channels
        model_cfg['in_chans'] = input_channels
        ## change any model config here
        ## load model and optionally weights
        model = TransFeX(model_cfg).cuda()
    elif cfg.model.name == 'unetencoder':
        model = UNetEncoder3D(input_channels, cfg.model.output_channels, f_maps=cfg.model.f_maps, levels=cfg.model.levels, multiplier=cfg.model.multiplier).cuda()
    elif cfg.model.name == 'lku':
        model = LKUNet(input_channels, cfg.model.output_channels, start_channel=cfg.model.f_maps[0], levels=cfg.model.levels).cuda()
    elif cfg.model.name == 'lkuencoder':
        model = LKUEncoder(input_channels, cfg.model.output_channels, start_channel=cfg.model.f_maps[0], levels=cfg.model.levels).cuda()
    elif cfg.model.name == 'lkuencoderv2':
        model = LKUEncoder(input_channels, cfg.model.output_channels, start_channel=cfg.model.f_maps[0], levels=cfg.model.levels, v2=True).cuda()
    else:
        raise ValueError(f"Unknown model name: {cfg.model.name}")

    print(f"Model has {count_parameters(model)} parameters.")
    if cfg.model.init_zero_features:
        print("Initializing zero features for the model...")
        model.init_zero_features()
    # load model
    # print(os.getcwd())
    cfg.save_dir = 'saved_models/' + cfg.tag + '/' + cfg.exp_name + '/'
    cfg.model_path = cfg.save_dir + "best_dice_loss.pth"
    saved_data = torch.load(cfg.model_path)
    epoch = saved_data['epoch']
    model.load_state_dict(saved_data['model'], strict=True)
    print(f"Resuming training...\nLoaded model and optim from: {cfg.model_path} at epoch: {saved_data['epoch']}.")

    # choose optimization solver
    if cfg.diffopt.warp_type == 'diffeomorphic':
        print("Choosing diffeomorphic")
        diffopt_solver = multi_scale_diffeomorphic_solver
    elif cfg.diffopt.warp_type == 'freeform':
        print("Choosing freeform")
        diffopt_solver = multi_scale_warp_solver
    elif cfg.diffopt.warp_type == 'affine':
        print("Choosing affine")
        diffopt_solver = multi_scale_affine2d_solver
    else:
        raise ValueError(f"Unknown solver: {cfg.diffopt.solver}")
    
    model.eval()
    # load NCC and Dice losses here
    feature_loss_fn = _get_loss_function_factory(cfg.diffopt.feature_loss_fn, cfg, spatial_dims=3)
    dice_loss_fn = DiceLossWithLongLabels(min_label=1, max_label=train_dataset.max_label_index, intersection_only=False)
    # get gaussians for opt
    gaussian_grad = gaussian_1d(torch.tensor(cfg.diffopt.sigma_grad), truncated=2).cuda() if cfg.diffopt.sigma_grad > 0 else None
    gaussian_warp = gaussian_1d(torch.tensor(cfg.diffopt.sigma_warp), truncated=2).cuda() if cfg.diffopt.sigma_warp > 0 else None

    # keep track of global step
    # run iterations
    # losses_ncc = []
    # losses_dice = []
    # losses_dice_all = []
    # vol_all = []

    till_feature_idx = resolve_layer_idx(epoch, cfg)
    cur_levels = reversed(sorted(cfg.model.levels)) if cfg.model.levels is not None else reversed([2**i for i in range(5)])
    cur_levels = list(cur_levels)[:till_feature_idx+1]
    print("current levels", cur_levels)

    # save all results here
    results_dict = dict()

    pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
    with torch.no_grad():
        for it, batch in pbar:
            # Sample 2D slices out of the volumes
            B, C, H, W, D = batch['source_img'].shape
            fixed_img, moving_img = batch['source_img'].cuda(), batch['target_img'].cuda()
            # if labels are present
            if batch['source_label'] is not None:
                fixed_label, moving_label = batch['source_label'].cuda(), batch['target_label'].cuda()
            else:
                fixed_label, moving_label = None, None
            
            # get features
            fixed_features, moving_features = model(fixed_img), model(moving_img)
            # get index to keep
            fixed_features = fixed_features[:till_feature_idx+1]
            moving_features = moving_features[:till_feature_idx+1]

            iterations = cfg.diffopt.iterations[:till_feature_idx+1]
            if cfg.diffopt.gradual:
                # in this case, we will see which one to use gradually
                if till_feature_idx == 0:
                    iterations[-1] = min((epoch + 1)*10, iterations[-1])
                else:
                    # only the last level is "gradualized"
                    iterations[-1] = min((epoch - cfg.train.train_new_level[till_feature_idx-1] + 1)*10, iterations[-1])

            ## Run multi-scale optimization
            if it == 0:
                print(iterations)
            # iterations = [0]*len(iterations)
            with torch.set_grad_enabled(True):
                displacements, losses_opt, jacnorm = diffopt_solver(fixed_features, moving_features,
                                    iterations=iterations, loss_function=feature_loss_fn, 
                                    phantom_step=cfg.diffopt.phantom_step, n_phantom_steps=0,
                                    debug=True,
                                    gaussian_grad=gaussian_grad, gaussian_warp=gaussian_warp)

            # For diffeomorphic or freeform, the displacements need to be converted to warps, for affine its already the warp
            if cfg.diffopt.warp_type in ['affine']:
                warps = displacements
            else:
                warps = displacements_to_warps(displacements)
            ## Get all losses ready 
            fixed_id = batch['source_img_id'][0].item()
            moving_id = batch['target_img_id'][0].item()
            # get final warp
            final_warp = warps[-1]
            print(final_warp.shape, final_warp.min(), final_warp.max())
            loss_dice = dice_loss_fn(moving_label, fixed_label, final_warp, train=False)
            dice_score = 1 - torch.stack(loss_dice).cpu().numpy()
            print(np.mean(dice_score), dice_score.shape)
            results_dict[(fixed_id, moving_id)] = {'dice': dice_score}

    # save
    # print(f"Saving to {cfg.save_dir}...")
    isotropic = "isotropic" if isotropic else "anisotropic"
    crop = "crop" if crop else "nocrop"
    print(np.mean([x['dice'] for x in results_dict.values()]))
    print(osp.join(cfg.save_dir, f'results_{dataset_name}_{isotropic}_{crop}.pkl'))
    with open(osp.join(cfg.save_dir, f'results_{dataset_name}_{isotropic}_{crop}.pkl'), 'wb') as f:
        pkl.dump(results_dict, f)
    
    print("\n"*5)

    # cleanup logging and wandb
    # cleanup(cfg, fp)
    

if __name__ == '__main__':
    mainfunc()
