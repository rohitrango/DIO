''' Copied from `train_single_level.py` but meant for 2D images for better visualization and debugging '''
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 
# make sure the parent of this folder is in path to be 
# able to access everything
from models.TransMorph import TransFeX
from models.unet3d import UNet2D
from models.configs_TransMorph import get_3DTransFeX_config
from solver.diffeo import multi_scale_warp_solver, multi_scale_diffeomorphic_solver
from solver.affine import multi_scale_affine2d_solver
from solver.utils import gaussian_1d, img2v_2d, v2img_2d, separable_filtering
from solver.losses import NCC_vxm, DiceLossWithLongLabels, _get_loss_function_factory
from solver.losses import LocalNormalizedCrossCorrelationLoss
from omegaconf import OmegaConf
from matplotlib import pyplot as plt

# logging
import wandb
import hydra
from model_utils import displacements_to_warps, downsample
from utils import set_seed, init_wandb, open_log, cleanup
from datasets.oasis import OASIS, OASISNeurite2D
import numpy as np

datasets = {
    'oasis': OASIS,
    'neurite-oasis': OASISNeurite2D
}

from train_multi_level_2d import clip_grad_2d, resolve_layer_idx, try_retaingrad, torch2wandbimg

@hydra.main(config_path='../../configs/dio/', config_name='oasis_ml_freeform_d4_2D')
def main(cfg):
    # init setup
    # override 
    cfg.deploy = False
    print(OmegaConf.to_yaml(cfg))
    set_seed(cfg.seed)

    # load dataset
    dataset_name = cfg.dataset.name
    val_dataset = datasets[dataset_name](cfg.dataset.data_root, split='train')  # oasis val doesnt have labels
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    print(f"Dataset has {len(val_dataset)} samples.")

    ### Load UNet with no skip connections
    model = UNet2D(1, cfg.model.output_channels, levels=cfg.model.levels, skip=cfg.model.skip).cuda()
    if cfg.model.init_zero_features:
        print("Initializing zero features for the model...")
        model.init_zero_features()
    # load the model from saved path
    cfg.model_path = 'saved_models/' + cfg.tag + '/' + cfg.exp_name + '/' + "best_dice_loss.pth"
    saved_data = torch.load(cfg.model_path)
    model.load_state_dict(saved_data['model'], strict=True)
    epoch = saved_data['epoch']
    # either resume training or just load from existing model
    print(f"Testing...\nLoaded model and optim from: {cfg.model_path} at epoch: {saved_data['epoch']}.")
    
    # choose optimization solver
    if cfg.diffopt.warp_type == 'diffeomorphic':
        diffopt_solver = multi_scale_diffeomorphic_solver
        # raise NotImplementedError("Diffeomorphic solver not implemented yet.")
    elif cfg.diffopt.warp_type == 'freeform':
        diffopt_solver = multi_scale_warp_solver
    elif cfg.diffopt.warp_type == 'affine':
        diffopt_solver = multi_scale_affine2d_solver
    else:
        raise ValueError(f"Unknown solver: {cfg.diffopt.solver}")
    
    model.eval()
    # load NCC and Dice losses here
    # feature_loss_fn = _get_loss_function_factory(cfg.diffopt.feature_loss_fn, cfg, spatial_dims=2)
    dice_loss_fn = DiceLossWithLongLabels(min_label=1, max_label=val_dataset.max_label_index, intersection_only=True)
    # get gaussians for opt
    gaussian_grad = gaussian_1d(torch.tensor(cfg.diffopt.sigma_grad), truncated=2).cuda() if cfg.diffopt.sigma_grad > 0 else None
    gaussian_warp = gaussian_1d(torch.tensor(cfg.diffopt.sigma_warp), truncated=2).cuda() if cfg.diffopt.sigma_warp > 0 else None

    # keep track of global step
        # run iterations
    till_feature_idx = resolve_layer_idx(epoch, cfg)
    cur_levels = reversed(sorted(cfg.model.levels)) if cfg.model.levels is not None else reversed([2**i for i in range(5)])
    cur_levels = list(cur_levels)[:till_feature_idx+1]
    print("Current levels: ", cur_levels)

    # keep track of dice scores
    dice_scores_all = []

    with torch.no_grad():
        for it, batch in enumerate(val_dataloader):
            # Sample 2D slices out of the volumes
            B, C, H, W, D = batch['source_img'].shape
            slice_id = D//2 
            fixed_img, moving_img = batch['source_img'][:, :, :, :, slice_id].cuda(), batch['target_img'][:, :, :, :, slice_id].cuda()
            fixed_label, moving_label = batch['source_label'][:, :, :, :, slice_id].cuda(), batch['target_label'][:, :, :, :, slice_id].cuda()
            fixed_img, moving_img, fixed_label, moving_label = [x.transpose(2, 3) for x in [fixed_img, moving_img, fixed_label, moving_label]]
            
            # get features
            fixed_features, moving_features = model(fixed_img), model(moving_img)
            # get index to keep
            fixed_features = fixed_features[:till_feature_idx+1]
            moving_features = moving_features[:till_feature_idx+1]
            # get iterations
            iterations = cfg.diffopt.iterations[:till_feature_idx+1]
            if cfg.diffopt.gradual:
                # in this case, we will see which one to use gradually
                if till_feature_idx == 0:
                    iterations[-1] = min((epoch + 1)*10, iterations[-1])
                else:
                    # only the last level is "gradualized"
                    iterations[-1] = min((epoch - cfg.train.train_new_level[till_feature_idx-1] + 1)*10, iterations[-1])
            ## Run multi-scale optimization
            with torch.set_grad_enabled(True):
                displacements, losses_opt, jacnorm = diffopt_solver(fixed_features, moving_features,
                                    # iterations=iterations, loss_function=feature_loss_fn, 
                                    iterations=iterations, loss_function=F.mse_loss, phantom_step=cfg.diffopt.phantom_step, n_phantom_steps=0,
                                    debug=True, 
                                    gaussian_grad=gaussian_grad, gaussian_warp=gaussian_warp)
            # print([len(x) for x in losses_opt])

            # For diffeomorphic or freeform, the displacements need to be converted to warps, for affine its already the warp
            # we just need the last warp
            if cfg.diffopt.warp_type in ['affine']:
                warps = displacements[-1:]
            else:
                warps = displacements_to_warps(displacements)[-1:]

            # len(warps) = 1
            # print(warps[0].shape)
            assert len(warps) == 1 and warps[0].shape[1:-1] == fixed_img.shape[2:], "Warp shape is not the same as the fixed image shape"
            for i in range(len(warps)):
                # warps[i].register_hook(lambda grad: v2img_2d(separable_filtering(img2v_2d(grad), gaussian_grad_back)))
                if cfg.loss.downsampled_warps:
                    ## we will downsample the fixed image and fixed label, and compute the ncc and dice losses 
                    # this is not the original scale
                    if warps[i].shape[1:-1] != fixed_img.shape[2:]:
                        fixed_img_down = downsample(fixed_img, size=fixed_features[i].shape[2:], mode='bilinear')
                        moving_img_down = downsample(moving_img, size=fixed_features[i].shape[2:], mode='bilinear')
                        fixed_label_down = F.interpolate(fixed_label.float(), size=fixed_features[i].shape[2:], mode='nearest').long()
                        moving_label_down = F.interpolate(moving_label.float(), size=fixed_features[i].shape[2:], mode='nearest').long()
                    else:
                        fixed_img_down, fixed_label_down = fixed_img, fixed_label
                        moving_img_down, moving_label_down = moving_img, moving_label
                    # compute NCC
                    moved_img = F.grid_sample(moving_img_down, warps[i], align_corners=True)
                    # compute losses
                    loss_dice = dice_loss_fn(moving_label_down, fixed_label_down, warps[i], train=False)
                else:
                    ## Downsampled warps is False, we upsample the warp field instead, and compute the dice and ncc losses
                    if warps[i].shape[1:-1] != fixed_img.shape[2:]:
                        warps[i] = img2v_2d(F.upsample(v2img_2d(warps[i]), size=fixed_img.shape[2:], mode='bilinear', align_corners=True))
                    moved_img = F.grid_sample(moving_img, warps[i], align_corners=True)
                    loss_dice = dice_loss_fn(moving_label, fixed_label, warps[i], train=False)

                # moved_label = F.grid_sample(moving_label.float(), warps[i], mode='nearest', align_corners=True).long()
                # # plot
                # plt.clf()
                # fig, ax = plt.subplots(1, 3, figsize=(15, 5))
                # ax[0].imshow(moving_img[0, 0].cpu().numpy(), cmap='gray'); ax[0].set_title('Moving Image')
                # ax[1].imshow(fixed_img[0, 0].cpu().numpy(), cmap='gray'); ax[1].set_title('Fixed Image')
                # ax[2].imshow(moved_img[0, 0].cpu().numpy(), cmap='gray'); ax[2].set_title('Moved Image')
                # [a.axis('off') for a in ax.ravel()]
                # plt.savefig('warps_{}.png'.format(it % 10), bbox_inches='tight')

                # fig, ax = plt.subplots(1, 3, figsize=(15, 5))
                # ax[0].imshow(moving_label[0, 0].cpu().numpy(),); ax[0].set_title('Moving Label')
                # ax[1].imshow(fixed_label[0, 0].cpu().numpy(),); ax[1].set_title('Fixed Label')
                # ax[2].imshow(moved_label[0, 0].cpu().numpy(),); ax[2].set_title('Moved Label')
                # [a.axis('off') for a in ax.ravel()]
                # plt.savefig('warps_label_{}.png'.format(it % 10), bbox_inches='tight')
                # plt.close('all')

                # warp = warps[i] 
                # print(warp.min(), warp.max(), warp.abs().mean())
                mean_dice_val = 1 - torch.mean(torch.stack(loss_dice)).item()
                dice_scores_all.append(mean_dice_val)
                # append to the ncc and dice losses
            
            print("# processed images: {}/{}, mean dice score: {}, current dice score: {}".format(it+1, len(val_dataloader), np.mean(dice_scores_all), mean_dice_val))

    # cleanup logging and wandb
    # cleanup(cfg, fp)
    

if __name__ == '__main__':
    main()
