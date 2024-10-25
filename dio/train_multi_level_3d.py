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
from TransMorph.models.unet3d import UNet2D, UNet3D, UNetEncoder3D
from TransMorph.models.lku import LKUNet, LKUEncoder
from TransMorph.models.configs_TransMorph import get_3DTransFeX_config
from solver.adam import multi_scale_warp_solver, multi_scale_diffeomorphic_solver, multi_scale_affine2d_solver
from solver.utils import gaussian_1d, img2v_3d, v2img_3d, separable_filtering
from solver.losses import NCC_vxm, DiceLossWithLongLabels, _get_loss_function_factory
from solver.losses import LocalNormalizedCrossCorrelationLoss
from omegaconf import OmegaConf
# logging
import wandb
import hydra
from model_utils import displacements_to_warps, downsample
from utils import set_seed, init_wandb, open_log, cleanup
from datasets.oasis import OASIS, OASISNeurite3D
import numpy as np
from scipy.ndimage import zoom
# torch.set_num_threads(1)
import argparse
import datetime

# ddp
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# this is a global setting to stay compatible with scipy's grid sampling
from solver.adam import ALIGN_CORNERS as align_corners

datasets = {
    'oasis': OASIS,
    'neurite-oasis': OASISNeurite3D,
}

def resolve_layer_idx(epoch, cfg):
    ''' resolve the layer index to use for the current epoch '''
    epoch_new_level = cfg.train.train_new_level
    idx = 0
    for lvl in epoch_new_level:
        if epoch >= lvl:
            idx += 1
    return idx

def try_retaingrad(tensors):
    try:
        for tensor in tensors:
            tensor.retain_grad()
    except:
        pass

def torch2wandbimg(tensor, mask_data=None):
    # convert to numpy and optionally mask
    tensor_npy = tensor.detach().cpu().numpy()
    tensor_npy = (tensor_npy - tensor_npy.min()) / (tensor_npy.max() - tensor_npy.min())
    if mask_data is None:
        return wandb.Image(tensor_npy)
    # check if tensor and mask sizes are different
    return wandb.Image(tensor_npy, masks={'labels': {'mask_data': mask_data}})

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def setup_ddp(rank, world_size, port=12355):
    # setup ddp environment
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    # dist.init_process_group("gloo", rank=rank, world_size=world_size, timeout=datetime.timedelta(seconds=1000))

def cleanup_ddp():
    dist.destroy_process_group()

@hydra.main(config_path='./configs', config_name='default')
def mainfunc(cfg):
    # get ddp if specified
    if not cfg.ddp.enabled:
        main(0, cfg)
    else:
        # setup ddp
        world_size = cfg.ddp.world_size
        mp.spawn(main, args=(cfg, world_size), nprocs=world_size, join=True)

# main function for every process
def main(rank, cfg, world_size=1):
    # is ddp enabled
    ddp = cfg.ddp.enabled
    isrankzero = rank == 0
    if ddp:
        setup_ddp(rank, world_size, port=cfg.ddp.port)
        torch.cuda.set_device(rank)
    # init setup
    if cfg.loss.downsampled_label_warps is None:
        cfg.loss.downsampled_label_warps = cfg.loss.downsampled_warps
    # if combine fixed and moving image, then input channels is 2, and output channels will be doubled
    if cfg.model.combine:
        cfg.model.output_channels = cfg.model.output_channels * 2
        input_channels = 2
    else:
        input_channels = 1

    init_wandb(cfg, 'TransFeX', rank)
    set_seed(cfg.seed)
    print, fp = open_log(cfg, rank) 

    # Save config file
    if cfg.deploy and isrankzero:
        OmegaConf.save(cfg, os.path.join(cfg.save_dir, 'config.yaml'))

    # load dataset
    dataset_name = cfg.dataset.name
    train_dataset = datasets[dataset_name](cfg.dataset.data_root, split='train')
    val_dataset = datasets[dataset_name](cfg.dataset.data_root, split='val')
    # create dataloader with or without ddp
    if ddp:
        train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0, sampler=DistributedSampler(train_dataset,))
        val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, sampler=DistributedSampler(val_dataset))
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
        val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    # only be used by rank 0, so no need to ddp
    print(f"Dataset has {len(train_dataset)}, {len(val_dataset)} samples.")

    ### Load UNet with no skip connections
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
    # add ddp to model
    if ddp:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    if not cfg.deploy:
        print(model)
    print(f"Model has {count_parameters(model)} parameters.")
    if cfg.model.init_zero_features:
        print("Initializing zero features for the model...")
        model.init_zero_features()

    optim = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, amsgrad=cfg.model.name == 'transmorph')
    # scheduler = torch.optim.lr_scheduler.PolynomialLR(optim, cfg.train.epochs, power=cfg.train.lr_power_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lambda epoch: (1 - epoch/cfg.train.epochs)**cfg.train.lr_power_decay)
    start_epoch = 0
    best_dice_loss = np.inf
    best_image_loss = np.inf
    # either resume training or just load from existing model
    if cfg.resume: 
        if cfg.model_path is not None:
            saved_data = torch.load(cfg.model_path)
        else:
            print("Could not load model from the specified path, trying to load from the save_dir")
            saved_data = torch.load(os.path.join(cfg.save_dir, 'best_dice_loss.pth'))
        if ddp:
            model.module.load_state_dict(saved_data['model'])
        else:
            model.load_state_dict(saved_data['model'])
        optim.load_state_dict(saved_data['optim'])
        try:
            scheduler.load_state_dict(saved_data['scheduler'])
        except:
            print("Could not load scheduler state, using default scheduler")
        print(f"Resuming training...\nLoaded model and optim from: {cfg.model_path} at epoch: {saved_data['epoch']}.")
        # load other metrics
        start_epoch = saved_data['epoch'] + 1
        best_dice_loss = saved_data['best_dice_loss']
        best_image_loss = saved_data['best_image_loss']
        print("Best Dice loss: %.6f, Best Image loss: %.6f" % (best_dice_loss, best_image_loss))
    elif cfg.model_path is not None:
        saved_data = torch.load(cfg.model_path)
        if ddp:
            model.module.load_state_dict(saved_data['model'])
        else:
            model.load_state_dict(saved_data['model'])
        print(f"Loaded pretrained model from: {cfg.model_path}.")
    else:
        print("Training from scratch...")

    if ddp:
        dist.barrier()
    
    # choose optimization solver
    if cfg.diffopt.warp_type == 'diffeomorphic':
        diffopt_solver = multi_scale_diffeomorphic_solver
    elif cfg.diffopt.warp_type == 'freeform':
        diffopt_solver = multi_scale_warp_solver
    elif cfg.diffopt.warp_type == 'affine':
        diffopt_solver = multi_scale_affine2d_solver
    else:
        raise ValueError(f"Unknown solver: {cfg.diffopt.solver}")
    
    model.train()
    # load NCC and Dice losses here
    feature_loss_fn = _get_loss_function_factory(cfg.diffopt.feature_loss_fn, cfg, spatial_dims=3)
    if cfg.loss.img_loss == 'ncc':
        ncc_img_loss_fn = LocalNormalizedCrossCorrelationLoss(spatial_dims=3, kernel_size=7, reduction='mean', 
                                                            image_noise_sigma=cfg.train.image_noise_sigma,
                                                            smooth_nr=cfg.train.cc_smooth_nr, unsigned=cfg.train.cc_unsigned)
    elif cfg.loss.img_loss == 'mse':
        ncc_img_loss_fn = lambda x, y: F.mse_loss(x, y) - 1  # to compensate for a +1 term later
    else:
        raise ValueError(f"Unknown image loss: {cfg.loss.img_loss}")

    dice_loss_fn = DiceLossWithLongLabels(min_label=1, max_label=train_dataset.max_label_index, intersection_only=True,
                                          order=cfg.loss.dice_order, center_dice_threshold=cfg.loss.center_dice_threshold if cfg.loss.weight_label_center > 0 else None, l2_mode=cfg.loss.dice_l2_mode)
    # label_center_fn = LabelCenterLoss(min_label=1, max_label=train_dataset.max_label_index, dice_threshold=cfg.loss.center_dice_threshold)
    # get gaussians for opt
    gaussian_grad = gaussian_1d(torch.tensor(cfg.diffopt.sigma_grad), truncated=2).cuda() if cfg.diffopt.sigma_grad > 0 else None
    gaussian_warp = gaussian_1d(torch.tensor(cfg.diffopt.sigma_warp), truncated=2).cuda() if cfg.diffopt.sigma_warp > 0 else None

    # keep track of global step
    global_step = start_epoch * len(train_dataloader)
    for epoch in range(start_epoch, cfg.train.epochs):
        # run iterations
        losses_ncc = []
        losses_dice = []
        till_feature_idx = resolve_layer_idx(epoch, cfg)
        cur_levels = reversed(sorted(cfg.model.levels)) if cfg.model.levels is not None else reversed([2**i for i in range(5)])
        cur_levels = list(cur_levels)[:till_feature_idx+1]
        if not cfg.deploy:
            print("current levels", cur_levels)

        iterations = cfg.diffopt.iterations[:till_feature_idx+1]
        if cfg.diffopt.gradual:
            # in this case, we will see which one to use gradually
            if till_feature_idx == 0:
                iterations[-1] = min((epoch + 1)*10, iterations[-1])
            else:
                # only the last level is "gradualized"
                iterations[-1] = min((epoch - cfg.train.train_new_level[till_feature_idx-1] + 1)*10, iterations[-1])

        model.train()
        for it, batch in enumerate(train_dataloader):
            optim.zero_grad()
            # Sample 2D slices out of the volumes
            B, C, H, W, D = batch['source_img'].shape
            fixed_img, moving_img = batch['source_img'].cuda(), batch['target_img'].cuda()
            fixed_label, moving_label = batch['source_label'].cuda(), batch['target_label'].cuda()
            
            # get features
            if cfg.model.combine:
                # combine, so only one model
                both_features = model(torch.cat([fixed_img, moving_img], dim=1))
                channels = [x.shape[1]//2 for x in both_features]
                fixed_features = [x[:, :c] for x, c in zip(both_features, channels)]
                moving_features = [x[:, c:] for x, c in zip(both_features, channels)]
            else:
                if cfg.train.backprop_both:
                    fixed_features, moving_features = model(fixed_img), model(moving_img)
                else:
                    U = np.random.rand() < 0.5
                    with torch.set_grad_enabled(U):
                        fixed_features = model(fixed_img)
                    with torch.set_grad_enabled(not U):
                        moving_features = model(moving_img)
                
            # print([f.shape for f in fixed_features], [f.shape for f in moving_features])
            # get index to keep
            dummy_ddp_loss = 0
            if ddp:
                dummy_ddp_loss = 0 * torch.stack([x.mean() for x in fixed_features]).mean() + 0 * torch.stack([x.mean() for x in moving_features]).mean()
                        
            fixed_features = fixed_features[:till_feature_idx+1]
            moving_features = moving_features[:till_feature_idx+1]

            # # retain grad of the feature images for debugging
            if not cfg.deploy:
                if it == 2:
                    break
                try_retaingrad(fixed_features)
                try_retaingrad(moving_features)  
                print([f.shape for f in fixed_features])
            
            ## Run multi-scale optimization
            displacements, losses_opt, jacnorm = diffopt_solver(fixed_features, moving_features,
                                iterations=iterations, loss_function=feature_loss_fn, 
                                phantom_step=cfg.diffopt.phantom_step, n_phantom_steps=cfg.diffopt.n_phantom_steps,
                                debug=True, convergence_eps=cfg.diffopt.convergence_eps,
                                gaussian_grad=gaussian_grad, gaussian_warp=gaussian_warp)
            if not cfg.deploy:
                print([len(x) for x in losses_opt])
            
            if cfg.loss.train_last_warp_only:
                displacements = displacements[-1:]
            # print(len(displacements))
            # Clip warp gradients if specified
            # if cfg.diffopt.clip_warp_grad_per < 100:
            #     for i, (disp, loss) in enumerate(zip(displacements, losses_opt)):
            #         # convergence did not happen for this level, clip gradients
            #         if len(losses_opt) < iterations[i] + 3:
            #             disp.register_hook(lambda grad: clip_grad_2d(grad, cfg.diffopt.clip_warp_grad_per))

            # For diffeomorphic or freeform, the displacements need to be converted to warps, for affine its already the warp
            if cfg.diffopt.warp_type in ['affine']:
                warps = displacements
            else:
                warps = displacements_to_warps(displacements)
            ## Get all losses ready 
            losses = {
                'ncc': [],
                'dice': [],
                'center': [],
            }
            ## Upscale all warps and compute losses
            # gaussian_grad_back = gaussian_1d(torch.tensor(1), truncated=2).cuda() 
            for i in range(len(warps)):
                # warps[i].register_hook(lambda grad: v2img_2d(separable_filtering(img2v_2d(grad), gaussian_grad_back)))
                if cfg.loss.downsampled_warps:
                    ## we will downsample the fixed image and fixed label, and compute the ncc and dice losses 
                    # this is not the original scale
                    if warps[i].shape[1:-1] != fixed_img.shape[2:]:
                        fixed_img_down = downsample(fixed_img, size=fixed_features[i].shape[2:], mode='trilinear')
                        moving_img_down = downsample(moving_img, size=fixed_features[i].shape[2:], mode='trilinear')
                    else:
                        fixed_img_down, moving_img_down = fixed_img, moving_img
                    # compute NCC
                    moved_img = F.grid_sample(moving_img_down, warps[i], align_corners=align_corners)
                    # compute losses
                    loss_ncc = 1+ncc_img_loss_fn(moved_img, fixed_img_down)
                else:
                    fixed_img_down, moving_img_down = fixed_img, moving_img
                    ## Downsampled warps is False, we upsample the warp field instead, and compute the dice and ncc losses
                    warp = warps[i]
                    if warps[i].shape[1:-1] != fixed_img.shape[2:]:
                        warp = img2v_3d(F.upsample(v2img_3d(warps[i]), size=fixed_img.shape[2:], mode='trilinear', align_corners=align_corners))
                    moved_img = F.grid_sample(moving_img, warp, align_corners=align_corners)
                    loss_ncc = 1+ncc_img_loss_fn(moved_img, fixed_img)

                # same for labels, also compute center loss here
                if cfg.loss.downsampled_label_warps:
                    if warps[i].shape[1:-1] != fixed_img.shape[2:]:
                        fixed_label_down = F.interpolate(fixed_label.float(), size=fixed_features[i].shape[2:], mode='nearest').long()
                        moving_label_down = F.interpolate(moving_label.float(), size=fixed_features[i].shape[2:], mode='nearest').long()
                    else:
                        fixed_label_down, moving_label_down = fixed_label, moving_label
                    # compute dice
                    loss_dice = dice_loss_fn(moving_label_down, fixed_label_down, warps[i])
                else:
                    ## Downsampled warps is False, we upsample the warp field instead, and compute the dice and ncc losses
                    fixed_label_down, moving_label_down = fixed_label, moving_label
                    warp = warps[i]
                    if warps[i].shape[1:-1] != fixed_img.shape[2:]:
                        warp = img2v_3d(F.upsample(v2img_3d(warps[i]), size=fixed_img.shape[2:], mode='trilinear', align_corners=align_corners))
                    loss_dice = dice_loss_fn(moving_label, fixed_label, warp)
                
                # From the dice loss function, get the center values 
                center_dists = dice_loss_fn.center_dist
                center_dice_scores = dice_loss_fn.center_dice_scores
                loss_center = torch.tensor(0.0, device=fixed_img.device)
                if len(center_dists) > 0:
                    loss_center = [(-score + dice_loss_fn.center_dice_threshold) * dist for score, dist in zip(center_dice_scores, center_dists)]
                    loss_center = torch.stack(loss_center).mean() / dice_loss_fn.center_dice_threshold

                mean_loss_dice = torch.mean(torch.stack(loss_dice))
                # append to the ncc and dice losses
                losses['ncc'].append(loss_ncc)
                losses['dice'].append(mean_loss_dice)
                losses['center'].append(loss_center)

            loss = dummy_ddp_loss
            if cfg.loss.weight_ncc > 0:
                loss = loss + cfg.loss.weight_ncc * torch.stack(losses['ncc']).mean()
            if cfg.loss.weight_dice > 0:
                loss = loss + cfg.loss.weight_dice * torch.stack(losses['dice']).mean()
            if cfg.loss.weight_jacobian > 0:
                loss = loss + cfg.loss.weight_jacobian * jacnorm.mean()
            if cfg.loss.weight_label_center > 0:
                loss = loss + cfg.loss.weight_label_center * torch.stack(losses['center']).mean()
            # add mse loss
            loss_mse = torch.tensor(0, device=loss.device)
            # decaying loss with epoch
            if cfg.loss.decay_mse > 0:
                moving_decoded_features = model.decode_features(moving_features)
                fixed_decoded_features = model.decode_features(fixed_features)
                for mov in moving_decoded_features:
                    loss_mse = loss_mse + F.mse_loss(F.interpolate(mov, size=moving_img.shape[2:], mode='bilinear', align_corners=align_corners), moving_img)
                for fix in fixed_decoded_features:
                    loss_mse = loss_mse + F.mse_loss(F.interpolate(fix, size=fixed_img.shape[2:], mode='bilinear', align_corners=align_corners), fixed_img)
                loss = loss + ((cfg.loss.decay_mse**epoch) * 0.5 * loss_mse / len(moving_decoded_features))
            # all 3 losses are added up now
            loss.backward()
            # change things to log
            # print details (or to log)
            print("Epoch: {}, Iter: {}, NCC: {}, Diceloss: {}, centerloss: {}, MSE: {:06f}, mse_lambda: {:06f}, lr: {:06f}, diffopt_iters: {}".format(
                        epoch, it, [np.around(x.item(), 6) for x in losses['ncc']], [np.around(x.item(), 6) for x in losses['dice']], [np.around(x.item(), 6) for x in losses['center']],
                                                            loss_mse.item(), (cfg.loss.decay_mse**epoch) , scheduler.get_last_lr()[0], iterations))
            # print the gradient values of the fixed and moving images for debugging
            if not cfg.deploy:
                norm_fixed_grad = torch.log10(fixed_features[-1].grad.norm()).item() if fixed_features[0].grad is not None else 0
                norm_mov_grad = torch.log10(moving_features[-1].grad.norm()).item() if moving_features[0].grad is not None else 0
                print("Fixed features grad norm: %.7f, Moving features grad norm: %.7f" % (norm_fixed_grad, norm_mov_grad))

            optim.step()
            global_step += 1
            # add losses to compute best model (based on last layer of warp)
            losses_ncc.append(loss_ncc.item())
            losses_dice.append(mean_loss_dice.item())
            # log statistics
            if cfg.deploy and cfg.log_wandb:
                # multi level loss opt
                # dont care about logging errors, just skip
                if isrankzero:
                    log_dict = {
                        'loss_ncc': loss_ncc.item(),
                        'loss_dice': mean_loss_dice.item(),
                        'loss_center': loss_center.item(),
                        'loss_mse': loss_mse.item(),
                        'norm Jacobian': jacnorm.item(),
                        'epoch': epoch,
                    }
                    losses_opt = [list(enumerate(losses)) for losses in losses_opt]
                    assert len(losses_opt) == len(cur_levels)
                    for lvl, loss_opt in zip(cur_levels, losses_opt):
                        log_dict[f'loss_opt_{lvl}'] = wandb.plot.line(wandb.Table(data=loss_opt, columns=['iter', 'loss']), "iter", "loss", title=f"Optimization Loss at level {lvl}")
                    # log_dict['warp_ncc_grad_norm'] = warp_ncc.grad.norm(dim=-1).mean().item()
                    # log_dict['warp_dice_grad_norm'] = warp_dice.grad.norm(dim=-1).mean().item()
                    if global_step % cfg.save_every == 0:
                        # add gradient images
                        B, _, Hi, Wi, Di = fixed_img_down.shape
                        B, C, Hf, Wf, Df = fixed_features[-1].shape
                        with torch.no_grad():
                            # if either labels or warps are processed at native resolution, use the native resolution only, and upsample warp 
                            if not cfg.loss.downsampled_warps or not cfg.loss.downsampled_label_warps:
                                fixed_img_down, moving_img_down = fixed_img, moving_img
                                fixed_label_down, moving_label_down = fixed_label, moving_label
                                warp = img2v_3d(F.upsample(v2img_3d(warps[-1]), size=fixed_img.shape[2:], mode='trilinear', align_corners=align_corners))
                                moved_image = F.grid_sample(moving_img, warp, align_corners=align_corners)
                                moved_label = F.grid_sample(moving_label.float(), warp, align_corners=align_corners, mode='nearest').long()
                                B, _, Hi, Wi, Di = fixed_img.shape
                                fixed_img_sample_idx = np.random.randint(0, 10)-5 + Di//2
                                # sample
                                fixed_img_sample, moving_img_sample = fixed_img[0, 0, :, :, fixed_img_sample_idx], moving_img[0, 0, :, :, fixed_img_sample_idx]
                                moved_img_sample = moved_image[0, 0, :, :, fixed_img_sample_idx].detach()
                                moved_label_sample = moved_label[0, 0, :, :, fixed_img_sample_idx].detach()
                            else:
                                fixed_img_sample_idx = np.random.randint(0, 10)-5 + Di//2
                                # sample the 2d slices
                                fixed_img_sample, moving_img_sample = fixed_img_down[0, 0, :, :, fixed_img_sample_idx], moving_img_down[0, 0, :, :, fixed_img_sample_idx]
                                moved_img_sample = moved_img[0, 0, :, :, fixed_img_sample_idx].detach()
                                # compute moved label 
                                moved_label = F.grid_sample(moving_label_down.float(), warps[-1], align_corners=align_corners, mode='nearest').long()
                                moved_label_sample = moved_label[0, 0, :, :, fixed_img_sample_idx].detach()
                            
                            # actually log the images
                            log_dict['fixed_img_mask'] = torch2wandbimg(fixed_img_sample, mask_data=fixed_label_down[0, 0, :, :, fixed_img_sample_idx].cpu().numpy())
                            log_dict['moving_img_mask'] = torch2wandbimg(moving_img_sample, mask_data=moving_label_down[0, 0, :, :, fixed_img_sample_idx].cpu().numpy())
                            log_dict['moved_img_mask'] = torch2wandbimg(moved_img_sample, mask_data=moved_label_sample.cpu().numpy())
                            log_dict['fixed_img'] = torch2wandbimg(fixed_img_sample)
                            log_dict['moving_img'] = torch2wandbimg(moving_img_sample)
                            log_dict['moved_img'] = torch2wandbimg(moved_img_sample)
                            # get label maps
                            fixed_img_sample_idx = np.random.randint(0, 10)-5 + Df//2
                            for c in range(C):
                                fixed_feature_sample, moving_feature_sample = fixed_features[-1][0, c, :, :, fixed_img_sample_idx], moving_features[-1][0, c, :, :, fixed_img_sample_idx]
                                log_dict[f'fixed_feature_{c}'] = torch2wandbimg(fixed_feature_sample)
                                log_dict[f'moving_feature_{c}'] = torch2wandbimg(moving_feature_sample)
                    wandb.log(log_dict)
                else:
                    pass

        if ddp:
            dist.barrier()
        # delete intermediates and make space for validation and next epoch
        # for f in fixed_features:
        #     del f
        # for m in moving_features:
        #     del m 
        # torch.cuda.empty_cache()
        ## validation loop
        model.eval()
        losses_ncc, losses_dice = [], []
        losses_dice_region = []
        with torch.no_grad():
            for it, batch in enumerate(val_dataloader):
                # dont do any validation if not rank 0
                # if not isrankzero:
                #     break
                if not cfg.deploy:
                    print(it, len(val_dataloader))
                # Sample 2D slices out of the volumes
                B, C, H, W, D = batch['source_img'].shape
                fixed_img, moving_img = batch['source_img'].cuda(), batch['target_img'].cuda()
                fixed_label, moving_label = batch['source_label'].cuda(), batch['target_label'].cuda()
                if cfg.model.combine:
                    both_features = model(torch.cat([fixed_img, moving_img], dim=1))
                    channels = [x.shape[1]//2 for x in both_features]
                    fixed_features = ([x[:, :c] for x, c in zip(both_features, channels)])[:till_feature_idx+1]
                    moving_features = ([x[:, c:] for x, c in zip(both_features, channels)])[:till_feature_idx+1]
                else:
                    fixed_features, moving_features = model(fixed_img), model(moving_img)
                    fixed_features, moving_features = fixed_features[:till_feature_idx+1], moving_features[:till_feature_idx+1]
                ## Run multi-scale optimization
                with torch.set_grad_enabled(True):
                    displacements, losses_opt, jacnorm = diffopt_solver(fixed_features, moving_features,
                                        iterations=iterations, loss_function=feature_loss_fn, 
                                        phantom_step=cfg.diffopt.phantom_step, n_phantom_steps=cfg.diffopt.n_phantom_steps,
                                        debug=True,
                                        gaussian_grad=gaussian_grad, gaussian_warp=gaussian_warp)
                if cfg.diffopt.warp_type in ['affine']:
                    warps = displacements
                else:
                    warps = displacements_to_warps(displacements)
                # given last warp, compute ncc and dice
                warp = warps[-1]
                if cfg.loss.downsampled_warps:
                    ## we will downsample the fixed image and fixed label, and compute the ncc and dice losses 
                    # this is not the original scale
                    if warp.shape[1:-1] != fixed_img.shape[2:]:
                        fixed_img_down = downsample(fixed_img, size=fixed_features[-1].shape[2:], mode='trilinear')
                        moving_img_down = downsample(moving_img, size=fixed_features[-1].shape[2:], mode='trilinear')
                    else:
                        fixed_img_down, moving_img_down = fixed_img, moving_img
                    # compute NCC
                    moved_img = F.grid_sample(moving_img_down, warp, align_corners=align_corners)
                    # compute losses
                    loss_ncc = 1+ncc_img_loss_fn(moved_img, fixed_img_down)
                else:
                    ## Downsampled warps is False, we upsample the warp field instead, and compute the dice and ncc losses
                    warpf = warp
                    if warp.shape[1:-1] != fixed_img.shape[2:]:
                        warpf = img2v_3d(F.upsample(v2img_3d(warp), size=fixed_img.shape[2:], mode='trilinear', align_corners=align_corners))
                    moved_img = F.grid_sample(moving_img, warpf, align_corners=align_corners)
                    loss_ncc = 1+ncc_img_loss_fn(moved_img, fixed_img)
                # Get dice loss 
                if cfg.loss.downsampled_label_warps:
                    if warp.shape[1:-1] != fixed_img.shape[2:]:
                        fixed_label_down = F.interpolate(fixed_label.float(), size=fixed_features[-1].shape[2:], mode='nearest').long()
                        moving_label_down = F.interpolate(moving_label.float(), size=fixed_features[-1].shape[2:], mode='nearest').long()
                    else:
                        fixed_label_down, moving_label_down = fixed_label, moving_label
                    # compute dice
                    loss_dice = dice_loss_fn(moving_label_down, fixed_label_down, warp, train=False)
                else:
                    ## Downsampled warps is False, we upsample the warp field instead, and compute the dice and ncc losses
                    warpf = warp
                    if warp.shape[1:-1] != fixed_img.shape[2:]:
                        warpf = img2v_3d(F.upsample(v2img_3d(warp), size=fixed_img.shape[2:], mode='trilinear', align_corners=align_corners))
                    loss_dice = dice_loss_fn(moving_label, fixed_label, warpf, train=False)
                # compute losses
                mean_loss_dice = torch.mean(torch.stack(loss_dice))
                losses_ncc.append(loss_ncc.item())
                losses_dice.append(mean_loss_dice.item())
                losses_dice_region.append([x.item() for x in loss_dice])

        # Ran validation, now log the results
        # if ddp, gather results from other ranks
        if ddp:
            dist.barrier()  # let everyone finish their validation
            losses_ncc_ddp = [None]*world_size
            losses_dice_ddp = [None]*world_size
            losses_dice_region_ddp = [None]*world_size
            dist.all_gather_object(losses_ncc_ddp, (losses_ncc, rank))
            dist.all_gather_object(losses_dice_ddp, (losses_dice, rank))
            dist.all_gather_object(losses_dice_region_ddp, (losses_dice_region, rank))
            if rank == 0:
                # sort objects
                losses_ncc_ddp = sorted(losses_ncc_ddp, key=lambda x: x[1])
                losses_dice_ddp = sorted(losses_dice_ddp, key=lambda x: x[1])
                losses_dice_region_ddp = sorted(losses_dice_region_ddp, key=lambda x: x[1])
                # get the actual values
                losses_ncc = [item for sublist, _ in losses_ncc_ddp for item in sublist][:19]
                losses_dice = [item for sublist, _ in losses_dice_ddp for item in sublist][:19]
                losses_dice_region = [item for sublist, _ in losses_dice_region_ddp for item in sublist][:19]
                # print(losses_ncc[0], losses_ncc[-1])
                # print(losses_dice[0], losses_dice[-1])
                # print(len(losses_ncc), len(losses_dice))
            # gather
            # losses_ncc = torch.tensor(losses_ncc).cuda()
            # losses_dice = torch.tensor(losses_dice).cuda()
            # losses_dice_region = torch.tensor(losses_dice_region).cuda()
            # dist.all_reduce(losses_ncc, op=dist.ReduceOp.SUM)
            # dist.all_reduce(losses_dice, op=dist.ReduceOp.SUM)
            # dist.all_reduce(losses_dice_region, op=dist.ReduceOp.SUM)
            # losses_ncc = losses_ncc.cpu().numpy()
            # losses_dice = losses_dice.cpu().numpy()
            # losses_dice_region = losses_dice_region.cpu().numpy()

        # log validation statistics
        if cfg.deploy and cfg.log_wandb and isrankzero:
            losses_dice_region = np.array(losses_dice_region).mean(0)  # [C]
            C = len(losses_dice_region)
            ret = {
                'val_loss_ncc': np.mean(losses_ncc),
                'val_loss_dice': np.mean(losses_dice),
                'val_epoch': epoch,
            }
            for i in range(C):
                ret[f'val_loss_dice_region_{i}'] = losses_dice_region[i]
            wandb.log(ret)
        
        # Print val scores
        if isrankzero:
            print("------ Validation Scores ------")
            print("Epoch: {}".format(epoch))
            print("Mean dice: {:.6f}".format(np.mean(losses_dice)))
            print("Mean ncc: {:.6f}".format(np.mean(losses_ncc)))
            print()

        # scheduler step
        scheduler.step()
        if cfg.deploy and isrankzero:
            # save best model according to dice loss
            if np.mean(losses_dice) < best_dice_loss:
                best_dice_loss = np.mean(losses_dice)
                torch.save({
                    'model': model.state_dict() if not ddp else model.module.state_dict(),
                    'optim': optim.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch,
                    'best_dice_loss': best_dice_loss,
                    'best_image_loss': np.mean(losses_ncc)
                }, os.path.join(cfg.save_dir, 'best_dice_loss.pth'))
            # save best model according to image loss
            if np.mean(losses_ncc) < best_image_loss:
                best_image_loss = np.mean(losses_ncc)
                torch.save({
                    'model': model.state_dict() if not ddp else model.module.state_dict(),
                    'optim': optim.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch,
                    'best_dice_loss': np.mean(losses_dice),
                    'best_image_loss': best_image_loss
                }, os.path.join(cfg.save_dir, 'best_image_loss.pth'))
        
        if ddp:
            dist.barrier()
    # cleanup logging and wandb
    cleanup(cfg, fp, rank)
    if ddp:
        cleanup_ddp()
    

if __name__ == '__main__':
    # try:
    mainfunc()
    # except:
    #     sys.stdout = sys.__stdout__
    #     sys.stderr = sys.__stderr__
        # wandb.finish()
