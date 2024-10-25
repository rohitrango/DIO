''' Copied from `train_single_level.py` but meant for 2D images for better visualization and debugging '''
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import sys
import os
# make sure the parent of this folder is in path to be 
# able to access everything
from models.TransMorph import TransFeX
from models.unet3d import UNet2D, UNet3D
from models.configs_TransMorph import get_3DTransFeX_config
from solver.diffeo import multi_scale_warp_solver, multi_scale_diffeomorphic_solver, multi_scale_affine2d_solver
from solver.utils import gaussian_1d, img2v_3d, v2img_3d, separable_filtering
from solver.losses import NCC_vxm, DiceLossWithLongLabels, _get_loss_function_factory
from solver.losses import LocalNormalizedCrossCorrelationLoss
from omegaconf import OmegaConf
from models.unet3d import UNet2D, UNet3D, UNetEncoder3D
from models.lku import LKUNet, LKUEncoder
from tqdm import tqdm
import nibabel as nib
from os import path as osp
from scipy.ndimage import gaussian_filter, zoom
from pprint import pprint
import pickle as pkl
from time import time

# logging
import wandb
import hydra
from model_utils import displacements_to_warps, downsample
from utils import set_seed, init_wandb, open_log, cleanup
from datasets.oasis import OASIS, OASISNeurite3D
import numpy as np
from evalutils import compute_metrics

datasets = {
    'oasis': OASIS,
    'neurite-oasis': OASISNeurite3D
}

from train_multi_level_3d import resolve_layer_idx, try_retaingrad, torch2wandbimg, count_parameters

def compute_detJac(warp):
    # warp : [B, H, W, D, 3]
    B = warp.shape[0]
    H, W, D = warp.shape[1:-1]
    dx = torch.from_numpy(np.array([[-1, 0, 1]])).float()[None, None, None].cuda()  # [1, 1, 1, 1, 3]
    dy = dx.transpose(3, 4).clone()  # [1, 1, 1, 3, 1]
    dz = dx.transpose(2, 4).clone()  # [1, 1, 3, 1, 1]
    # scale dx and dy
    dx = dx / (4/(H))
    dy = dy / (4/(W))
    dz = dz / (4/(D))
    jac = torch.zeros((B, H, W, D, 3, 3), device=warp.device)
    dxs = [dx, dy, dz]
    paddings = [(0,0,1), (0,1,0), (1,0,0)]
    for fi in range(3):
        for di in range(3):
            jac[..., fi, di] = F.conv3d(warp[:, None, ..., fi], dxs[di], padding=paddings[di])
    det = torch.linalg.det(jac)
    return det[...,1:-1, 1:-1, 1:-1]


@hydra.main(config_path='../configs/dio', config_name='oasis_ml_freeform_d4_3D')
def main(cfg):
    # init setup
    # init_wandb(cfg, project_name='TransFeX')
    cfg.deploy = False
    set_seed(cfg.seed)
    fp = open_log(cfg)

    # load dataset
    _ = cfg.learn2reg_eval
    dataset_name = cfg.dataset.name
    split = cfg.dataset.split
    assert split in ['val', 'test'], f"Unknown split in test script: {split}"
    train_dataset = datasets[dataset_name](cfg.dataset.data_root, split=split) 
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    print(f"Dataset has {len(train_dataset)} samples.")

    ### Load UNet with no skip connections
    # model = UNet3D(1, cfg.model.output_channels, levels=cfg.model.levels, skip=cfg.model.skip).cuda()
    # model = UNet3D(1, cfg.model.output_channels, f_maps=cfg.model.f_maps, levels=cfg.model.levels, skip=cfg.model.skip).cuda()
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

    # print(f"Model has {count_parameters(model)} parameters.")
    if cfg.model.init_zero_features:
        print("Initializing zero features for the model...")
        model.init_zero_features()
    # load model
    cfg.save_dir = 'saved_models/' + cfg.tag + '/' + cfg.exp_name + '/'
    cfg.model_path = cfg.save_dir + "best_dice_loss.pth"
    saved_data = torch.load(cfg.model_path)
    epoch = saved_data['epoch']
    model.load_state_dict(saved_data['model'])
    print(f"Resuming training...\nLoaded model and optim from: {cfg.model_path} at epoch: {saved_data['epoch']}.")
    print("Best dice score: ", 1-saved_data['best_dice_loss'])
    print(f"Model has {count_parameters(model)} parameters.")


    # choose optimization solver
    if cfg.diffopt.warp_type == 'diffeomorphic':
        diffopt_solver = multi_scale_diffeomorphic_solver
    elif cfg.diffopt.warp_type == 'freeform':
        diffopt_solver = multi_scale_warp_solver
    elif cfg.diffopt.warp_type == 'affine':
        diffopt_solver = multi_scale_affine2d_solver
    else:
        raise ValueError(f"Unknown solver: {cfg.diffopt.solver}")
    
    model.eval()
    # load NCC and Dice losses here
    feature_loss_fn = _get_loss_function_factory(cfg.diffopt.feature_loss_fn, cfg, spatial_dims=3)
    ncc_img_loss_fn = LocalNormalizedCrossCorrelationLoss(spatial_dims=3, kernel_size=7, reduction='mean', 
                                                          image_noise_sigma=cfg.train.image_noise_sigma,
                                                          smooth_nr=cfg.train.cc_smooth_nr, unsigned=cfg.train.cc_unsigned)
    dice_loss_fn = DiceLossWithLongLabels(min_label=1, max_label=train_dataset.max_label_index, intersection_only=False)
    # get gaussians for opt
    gaussian_grad = gaussian_1d(torch.tensor(cfg.diffopt.sigma_grad), truncated=2).cuda() if cfg.diffopt.sigma_grad > 0 else None
    gaussian_warp = gaussian_1d(torch.tensor(cfg.diffopt.sigma_warp), truncated=2).cuda() if cfg.diffopt.sigma_warp > 0 else None

    # keep track of global step
    # run iterations
    losses_ncc = []
    losses_dice = []
    losses_dice_all = []
    vol_all = []

    till_feature_idx = resolve_layer_idx(epoch, cfg)
    cur_levels = reversed(sorted(cfg.model.levels)) if cfg.model.levels is not None else reversed([2**i for i in range(5)])
    cur_levels = list(cur_levels)[:till_feature_idx+1]
    print("current levels", cur_levels)

    # save results
    all_results = dict()

    # save times
    all_times = dict()

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
            t1 = time()
            fixed_features, moving_features = model(fixed_img), model(moving_img)
            feature_time = time() - t1
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
            with torch.set_grad_enabled(True):
                t1 = time()
                displacements, losses_opt, jacnorm = diffopt_solver(fixed_features, moving_features,
                                    iterations=iterations, loss_function=feature_loss_fn, 
                                    phantom_step=cfg.diffopt.phantom_step, n_phantom_steps=0,
                                    debug=True,
                                    gaussian_grad=gaussian_grad, gaussian_warp=gaussian_warp)
                opt_time = time() - t1

            # save times
            print(feature_time, opt_time)
            all_times[it] = (feature_time, opt_time)

            # For diffeomorphic or freeform, the displacements need to be converted to warps, for affine its already the warp
            if cfg.diffopt.warp_type in ['affine']:
                warps = displacements
            else:
                warps = displacements_to_warps(displacements)

            fixed_id = batch['source_img_path'][0].split('/')[-1].split('.')[0].split("_")[-2]
            moving_id = batch['target_img_path'][0].split('/')[-1].split('.')[0].split("_")[-2]
            # add offset
            fixed_id = "0" + str(int(fixed_id) - 395 + 438)
            moving_id = "0" + str(int(moving_id) - 395 + 438)
            # compute the dice score and NCC with final warp
            final_warp = warps[-1]

            # if learn2reg eval
            if cfg.learn2reg_eval:
                assert final_warp.shape[1:-1] == fixed_img.shape[2:], f"Final warp shape: {final_warp.shape[1:-1]} and fixed img shape: {fixed_img.shape[2:]}"
                moved_image = F.grid_sample(moving_img, final_warp, align_corners=True)
                loss_ncc = 1 + ncc_img_loss_fn(moved_image, fixed_img).item()
                # dice score if val split
                if split == 'val':
                    loss_dice = dice_loss_fn(moving_label, fixed_label, final_warp, train=False)
                    losses_dice_all.append(torch.stack(loss_dice).cpu().numpy())
                    mean_loss_dice = torch.mean(torch.stack(loss_dice)).item()
                    vol_fixed = []
                    for labid in range(1, train_dataset.max_label_index+1):
                        vol_fixed.append(torch.sum(fixed_label == labid).item())
                    vol_all.append(vol_fixed)
                else:
                    mean_loss_dice = 0
                # append them
                losses_ncc.append(loss_ncc)
                losses_dice.append(mean_loss_dice)
                descr = f"It: {it}/{len(train_dataloader)} | NCC_cur: {loss_ncc:.4f} | Dice_cur: {mean_loss_dice:.4f} | NCC_avg: {np.mean(losses_ncc):.4f} | Dice_avg: {np.mean(losses_dice):.4f}"
                pbar.set_description(descr)
                # save the displacements 
                disp = displacements[-1][0].cpu().numpy()   # [H, W, D, 3]
                disp = disp[..., ::-1] + 0   # invert axes from [xyz] to [zyx]
                H, W, D = disp.shape[:3]
                disp[..., 0] *= (H-1)/2
                disp[..., 1] *= (W-1)/2
                disp[..., 2] *= (D-1)/2
                # now the format is [XYZ, xyz], convert it into neurite format
                disp = (disp.transpose(0, 2, 1, 3))[..., [0, 2, 1]] + 0
                disp = disp[::-1, ::-1]
                # flip the displacements
                disp[..., 0] *= -1
                disp[..., 1] *= -1
                disp = np.ascontiguousarray(disp)
                # smooth it out
                # for i in range(3):
                #     disp[..., i] = gaussian_filter(disp[..., i], sigma=1)
                # disp = disp[::2, ::2, ::2]
                if split == 'test':
                    disp = disp.transpose(3, 0, 1, 2)
                    disp = np.ascontiguousarray(disp.astype(np.float16))
                else:
                    newdisp = np.zeros([x//2 for x in disp.shape[:3]] + [3])
                    for i in range(3):
                        newdisp[..., i] = zoom(disp[..., i], 0.5, order=2)
                    ## if the images are downsampled themselves
                    # disp = newdisp.transpose(3, 0, 1, 2) / 2
                    ## if the displacement map is upsampled
                    disp = newdisp.transpose(3, 0, 1, 2) 
                    disp = np.ascontiguousarray(disp.astype(np.float16))
                
                print(disp.shape)
                os.makedirs(osp.join(cfg.save_dir, cfg.dataset.split), exist_ok=True)
                np.savez(osp.join(cfg.save_dir, cfg.dataset.split, f'disp_{fixed_id}_{moving_id}.npz'), disp)
                # nib.save(nib.Nifti1Image(disp, np.eye(4)), osp.join(cfg.save_dir, cfg.dataset.split, f'disp_{fixed_id}_{moving_id}.nii.gz'))
                print("Saved displacements to: ", osp.join(cfg.save_dir, cfg.dataset.split, f'disp_{fixed_id}_{moving_id}.npz'))
            else:
                # compute dice, HD95 and |J|<0
                if moving_label is None:
                    break
                moving_label = F.one_hot(moving_label.long()[0], num_classes=train_dataset.max_label_index+1).permute(0, 4, 1, 2, 3).float().cuda()[:, 1:]
                fixed_label = F.one_hot(fixed_label.long()[0], num_classes=train_dataset.max_label_index+1).permute(0, 4, 1, 2, 3).float().cuda()[:, 1:]
                moved_label = (F.grid_sample(moving_label, final_warp, align_corners=True)>=0.5).float()
                # print(moved_label.shape, fixed_label.shape, moving_label.shape, final_warp.shape)
                metrics = compute_metrics(moved_label, fixed_label, final_warp, method="transfex")
                detJac = compute_detJac(final_warp)
                detJac = (detJac <= 0).float().mean()
                metrics['detJac'] = detJac.item()
                pprint({k: np.mean(v) for k, v in metrics.items()})
                all_results[(fixed_id, moving_id)] = metrics

                # save warp in pytorch format
                os.makedirs(osp.join(cfg.save_dir, cfg.dataset.split), exist_ok=True)
                np.savez(osp.join(cfg.save_dir, cfg.dataset.split, f'warp_{fixed_id}_{moving_id}.npz'), final_warp.cpu().numpy())
                print("Saved warp to: ", osp.join(cfg.save_dir, cfg.dataset.split, f'warp_{fixed_id}_{moving_id}.npz'))

    # Print times
    feature_times = [x[0] for x in all_times.values()]
    opt_times = [x[1] for x in all_times.values()]
    print("Feature times: ", np.mean(feature_times), np.std(feature_times))
    print("Opt times: ", np.mean(opt_times), np.std(opt_times))

    # save validation results for eval in paper
    if not cfg.learn2reg_eval and split == 'val':
        while True:
            savename = input("Enter save pickle name. ").replace(" ", "")
            # dont save if empty name
            if savename == "":
                break
            if not savename.endswith(".pkl"):
                savename += ".pkl"
            path = osp.join(cfg.save_dir, savename)
            if osp.exists(path):
                print("File already exists. Enter a new name.")
            else:
                break
        # save
        if savename != "":
            with open(osp.join(cfg.save_dir, savename), 'wb') as f:
                pkl.dump(all_results, f)
            print("Saved to ", osp.join(cfg.save_dir, savename))

        # save the results
        # np.savez(osp.join(cfg.save_dir, cfg.dataset.split, 'results.npz'), all_results)
        # print("Saved results to: ", osp.join(cfg.save_dir, cfg.dataset.split, 'results.npz'))

    # For val split, plot the details of dice 
    # losses_dice_all = np.array(losses_dice_all)
    # vol_all = np.array(vol_all)
    # np.savez("val_dice_losses.npz", losses_dice_all=np.array(losses_dice_all), vol_all=np.array(vol_all))

    # cleanup logging and wandb
    cleanup(cfg, fp)
    

if __name__ == '__main__':
    main()
