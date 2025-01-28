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

def try_retaingrad(tensor):
    try:
        tensor.retain_grad()
    except:
        pass

def torch2wandbimg(tensor, mask_data=None):
    tensor_npy = tensor.detach().cpu().numpy()
    tensor_npy = (tensor_npy - tensor_npy.min()) / (tensor_npy.max() - tensor_npy.min())
    if mask_data is None:
        return wandb.Image(tensor_npy)
    return wandb.Image(tensor_npy, masks={'labels': {'mask_data': mask_data}})

@hydra.main(config_path='../../configs/dio/', config_name='oasis_sl_d1')
def main(cfg):
    # init setup
    init_wandb(cfg, project_name='TransFeX')
    set_seed(cfg.seed)
    fp = open_log(cfg)

    # load dataset
    dataset_name = cfg.dataset.name
    train_dataset = datasets[dataset_name](cfg.dataset.data_root, split='train')
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    print(f"Dataset has {len(train_dataset)} samples.")

    ### Load UNet with no skip connections
    model = UNet2D(1, cfg.model.output_channels, levels=[cfg.model.levels], skip=False).cuda()
    optim = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    scheduler = torch.optim.lr_scheduler.PolynomialLR(optim, cfg.train.epochs, power=cfg.train.lr_power_decay)
    start_epoch = 0
    best_dice_loss = np.inf
    best_image_loss = np.inf
    # either resume training or just load from existing model
    if cfg.resume: 
        saved_data = torch.load(cfg.model_path)
        model.load_state_dict(saved_data['model'])
        optim.load_state_dict(saved_data['optim'])
        scheduler.load_state_dict(saved_data['scheduler'])
        print(f"Resuming training...\nLoaded model and optim from: {cfg.model_path} at epoch: {saved_data['epoch']}.")
        # load other metrics
        start_epoch = saved_data['epoch'] + 1
        best_dice_loss = saved_data['best_dice_loss']
        best_image_loss = saved_data['best_image_loss']
    elif cfg.model_path is not None:
        saved_data = torch.load(cfg.model_path)
        model.load_state_dict(saved_data['model'])
        print(f"Loaded pretrained model from: {cfg.model_path}.")
    else:
        print("Training from scratch...")
    
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
    
    model.train()
    # load NCC and Dice losses here
    ncc_loss_fn = _get_loss_function_factory(cfg.diffopt.feature_loss_fn, cfg, spatial_dims=2)
    ncc_img_loss_fn = LocalNormalizedCrossCorrelationLoss(spatial_dims=2, kernel_size=7, reduction='mean')
    dice_loss_fn = DiceLossWithLongLabels(min_label=1, max_label=train_dataset.max_label_index)
    # get gaussians for opt
    gaussian_grad = gaussian_1d(torch.tensor(cfg.diffopt.sigma_grad), truncated=2).cuda() if cfg.diffopt.sigma_grad > 0 else None
    gaussian_warp = gaussian_1d(torch.tensor(cfg.diffopt.sigma_warp), truncated=2).cuda() if cfg.diffopt.sigma_warp > 0 else None
    
    for epoch in range(start_epoch, cfg.train.epochs):
        # run iterations
        losses_ncc = []
        losses_dice = []
        for it, batch in enumerate(train_dataloader):
            optim.zero_grad()
            # Sample 2D slices out of the volumes
            B, C, H, W, D = batch['source_img'].shape
            slice_id = D//2 + np.random.randint(11) - 5
            fixed_img, moving_img = batch['source_img'][:, :, :, :, slice_id].cuda(), batch['target_img'][:, :, :, :, slice_id].cuda()
            fixed_label, moving_label = batch['source_label'][:, :, :, :, slice_id].cuda(), batch['target_label'][:, :, :, :, slice_id].cuda()
            fixed_img, moving_img, fixed_label, moving_label = [x.transpose(2, 3) for x in [fixed_img, moving_img, fixed_label, moving_label]]
            
            # get features
            if cfg.train.backprop_both:
                fixed_features, moving_features = model(fixed_img), model(moving_img)
            else:
                U = np.random.rand() < 0.5
                with torch.set_grad_enabled(U):
                    fixed_features = model(fixed_img)
                with torch.set_grad_enabled(not U):
                    moving_features = model(moving_img)
            
            # retain grad of the feature images for debugging
            if not cfg.deploy:
                try_retaingrad(fixed_features[-1])
                try_retaingrad(moving_features[-1])  
            
            # optionally scale the gradient norm of the feature images
            if cfg.train.feature_grad_norm > 0:
                for f in fixed_features + moving_features:
                    print(f.requires_grad)
                    if f.requires_grad:
                        f.register_hook(lambda grad: grad * cfg.train.feature_grad_norm / (grad.norm() + 1e-8))

            # print(f"Fixed features: {fixed_features[0].shape}, Moving features: {moving_features[0].shape}")
            # now get displacements (minimum of 10 iterations)
            iterations = min((epoch + 1)*10, cfg.diffopt.iterations) if cfg.diffopt.gradual else cfg.diffopt.iterations
            displacements, losses_opt = diffopt_solver(fixed_features, moving_features,
                                iterations=[iterations], loss_function=ncc_loss_fn, 
                                debug=True,
                                gaussian_grad=gaussian_grad, gaussian_warp=gaussian_warp)[:2]
            losses_opt = losses_opt[0]
            # For diffeomorphic or freeform, the displacements need to be converted to warps, for affine its already the warp
            if cfg.diffopt.warp_type in ['affine']:
                warp = displacements[0]
            else:
                warp = displacements_to_warps(displacements)[0] # only one level
            # upsample it to match the original image size
            if cfg.model.levels != 1:
                warp = img2v_2d(F.upsample(v2img_2d(warp), size=fixed_img.shape[2:], mode='bilinear', align_corners=True))
            # keep two copies to compute the gradient individually
            warp_ncc = warp.clone()
            warp_dice = warp.clone()
            gaussian_grad_back = gaussian_1d(torch.tensor(1), truncated=2).cuda() 
            warp_ncc.register_hook(lambda grad: v2img_2d(separable_filtering(img2v_2d(grad), gaussian_grad_back)))
            warp_dice.register_hook(lambda grad: v2img_2d(separable_filtering(img2v_2d(grad), gaussian_grad_back)))
            # retain gradients
            warp_ncc.retain_grad()
            warp_dice.retain_grad()

            ### DEPRECATED: Downsample fixed image and label to map the warp
            # warp.retain_grad()
            # with torch.no_grad():
            #     if cfg.model.levels != 1:
            #         N = cfg.model.levels
            #         fixed_img_down = downsample(fixed_img, size=fixed_features[0].shape[2:], mode='bilinear')
            #         fixed_label_down = F.interpolate(fixed_label.float(), size=fixed_features[0].shape[2:], mode='nearest').long()
            #     else:
            #         fixed_img_down = fixed_img
            #         fixed_label_down = fixed_label
                
            # apply loss
            moved_img = F.grid_sample(moving_img, warp_ncc, align_corners=True)
            loss_ncc = 1+ncc_img_loss_fn(moved_img, fixed_img)
            loss_dice = dice_loss_fn(moving_label, fixed_label, warp_dice)
            mean_loss_dice = torch.mean(torch.stack(loss_dice))
            # loss = cfg.loss.weight_ncc * loss_ncc + cfg.loss.weight_dice * mean_loss_dice
            loss = 0
            if cfg.loss.weight_ncc > 0:
                loss = loss + cfg.loss.weight_ncc * loss_ncc
            if cfg.loss.weight_dice > 0:
                loss = loss + cfg.loss.weight_dice * mean_loss_dice
            # add mse loss
            loss_mse = torch.tensor(0, device=loss.device)
            # decaying loss with epoch
            if cfg.loss.decay_mse > 0:
                loss_mse = F.mse_loss(F.interpolate(model.decode_features(moving_features)[0], size=moving_img.shape[2:], mode='bilinear', align_corners=True), moving_img) \
                    + F.mse_loss(F.interpolate(model.decode_features(fixed_features)[0], size=fixed_img.shape[2:], mode='bilinear', align_corners=True), fixed_img)
                loss = loss + (cfg.loss.decay_mse**epoch) * 0.5 * loss_mse
            loss.backward()
            # print(warp_ncc.grad.min(), warp_ncc.grad.max(), warp_ncc.grad.abs().mean())
            # print(warp_dice.grad.min(), warp_dice.grad.max(), warp_dice.grad.abs().mean())
            # print details (or to log)
            print("Epoch: %d, Iter: %d, NCC: %.4f, Diceloss: %.4f, MSE: %.4f, mse_lambda: %.4f, lr: %.4f, diffopt_iters: %d" % (epoch, it, loss_ncc.item(), mean_loss_dice.item(), \
                                                            loss_mse.item(), (cfg.loss.decay_mse**epoch) , scheduler.get_last_lr()[0], iterations))
            # print the gradient values of the fixed and moving images for debugging
            if not cfg.deploy:
                norm_fixed_grad = torch.log10(torch.abs(fixed_features[0].grad).mean()).item() if fixed_features[0].grad is not None else 0
                norm_mov_grad = torch.log10(torch.abs(moving_features[0].grad).mean()).item() if moving_features[0].grad is not None else 0
                print("Fixed features grad norm: %.7f, Moving features grad norm: %.7f" % (norm_fixed_grad, norm_mov_grad))

            optim.step()
            # add losses to compute best model
            losses_ncc.append(loss_ncc.item())
            losses_dice.append(mean_loss_dice.item())
            # log statistics
            if cfg.deploy:
                loss_opt_wandb = wandb.Table(data=list(enumerate(losses_opt)), columns=['iter', 'loss'])
                log_dict = {
                    'loss_ncc': loss_ncc.item(),
                    'loss_dice': mean_loss_dice.item(),
                    'loss_mse': loss_mse.item(),
                    'loss_opt': wandb.plot.line(loss_opt_wandb, "iter", "loss", title="Optimization Loss"),
                    'epoch': epoch,
                }
                log_dict['warp_ncc_grad_norm'] = warp_ncc.grad.norm(dim=-1).mean().item()
                log_dict['warp_dice_grad_norm'] = warp_dice.grad.norm(dim=-1).mean().item()
                if it % 100 == 0:
                    # add gradient images
                    log_dict['warp_ncc_grad'] = torch2wandbimg(warp_ncc.grad.norm(dim=-1, keepdim=False)[0])
                    log_dict['warp_dice_grad'] = torch2wandbimg(warp_dice.grad.norm(dim=-1, keepdim=False)[0])
                    # add feature images
                    B, _, Hi, Wi = fixed_img.shape
                    B, C, Hf, Wf = fixed_features[0].shape
                    with torch.no_grad():
                        # fixed_img_sample_idx = np.random.randint(0, 10)-5 + Hi//2
                        # fixed_img_sample, moving_img_sample = fixed_img[0, 0, fixed_img_sample_idx], moving_img[0, 0, fixed_img_sample_idx]
                        # fixed_feature_sample_idx = np.random.randint(0, 10)-5 + Hf//2
                        # fixed_img_sample, moving_img_sample = fixed_img[0, 0, fixed_img_sample_idx], moving_img[0, 0, fixed_img_sample_idx]
                        fixed_img_sample, moving_img_sample = fixed_img[0, 0], moving_img[0, 0]
                        moved_img_sample = moved_img[0, 0].detach()
                        moved_label = F.grid_sample(moving_label.float(), warp, align_corners=True, mode='nearest').long()
                        log_dict['fixed_img_mask'] = torch2wandbimg(fixed_img_sample, mask_data=fixed_label[0, 0].cpu().numpy())
                        log_dict['moving_img_mask'] = torch2wandbimg(moving_img_sample, mask_data=moving_label[0, 0].cpu().numpy())
                        log_dict['moved_img_mask'] = torch2wandbimg(moved_img_sample, mask_data=moved_label[0, 0].cpu().numpy())
                        log_dict['fixed_img'] = torch2wandbimg(fixed_img_sample)
                        log_dict['moving_img'] = torch2wandbimg(moving_img_sample)
                        log_dict['moved_img'] = torch2wandbimg(moved_img_sample)
                        # get label maps
                        for c in range(C):
                            fixed_feature_sample, moving_feature_sample = fixed_features[-1][0, c], moving_features[-1][0, c]
                            log_dict[f'fixed_feature_{c}'] = torch2wandbimg(fixed_feature_sample)
                            log_dict[f'moving_feature_{c}'] = torch2wandbimg(moving_feature_sample)
                wandb.log(log_dict)

        # scheduler step
        scheduler.step()
        if cfg.deploy:
            # save best model according to dice loss
            if np.mean(losses_dice) < best_dice_loss:
                best_dice_loss = np.mean(losses_dice)
                torch.save({
                    'model': model.state_dict(),
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
                    'model': model.state_dict(),
                    'optim': optim.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch,
                    'best_dice_loss': np.mean(losses_dice),
                    'best_image_loss': best_image_loss
                }, os.path.join(cfg.save_dir, 'best_image_loss.pth'))
    # cleanup logging and wandb
    cleanup(cfg, fp)
    

if __name__ == '__main__':
    try:
        main()
    except:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        wandb.finish()
