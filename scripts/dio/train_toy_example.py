from typing import Any
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
import sys
import os
import wandb
import hydra
from utils import set_seed, init_wandb, open_log, cleanup, resolve_layer_idx, reshape_util
from datasets.toyexamples import SquareDataset, Square3DDataset
from models.unet import ResNetUNet as UNet
from models.unet3d import UNet3D
import numpy as np
from solver.affine import run_affine_transform_2d, run_affine_transform_3d

datasets = {
    'square': SquareDataset,
    'square3d': Square3DDataset,
}

@hydra.main(config_path='../../configs/dio', config_name='square')
def main(cfg):
    # init setup
    init_wandb(cfg, project_name='TransFeX-toyexample')
    set_seed(cfg.seed)
    fp = open_log(cfg)
    # load dataset
    dataset_name = cfg.dataset.name
    train_dataset = datasets[dataset_name](split='train')
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    sz = train_dataset.img_size_actual
    print(f"Dataset has {len(train_dataset)} samples.")
    # load model
    # change any model config here
    # load model and optionally weights
    is3d = '3d' in dataset_name.lower()
    if is3d:
        model = UNet3D(1, cfg.model.output_channels, f_maps=8, num_levels=cfg.model.max_level).cuda()
    else:
        model = UNet(cfg.model.output_channels, max_level=cfg.model.max_level, skip=cfg.model.skip).cuda()
    optim = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, ) #weight_decay=1e-5)
    # scheduler = torch.optim.lr_scheduler.PolynomialLR(optim, cfg.train.epochs, power=cfg.train.lr_power_decay)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optim, cfg.train.epochs, power=cfg.train.lr_power_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lambda x: (1 - x/cfg.train.epochs)**cfg.train.lr_power_decay)
    start_epoch = 0
    best_dice_loss = np.inf
    assert cfg.model.max_level == len(cfg.train.train_new_level)+1
    # either resume training or just load from existing model
    if cfg.resume: 
        saved_data = torch.load(cfg.model_path)
        model.load_state_dict(saved_data['model'])
        optim.load_state_dict(saved_data['optim'])
        # scheduler.load_state_dict(saved_data['scheduler'])
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
    
    model.train()
    # load NCC and Dice losses here
    def dice_loss_fn(pred, gt, eps=1e-5):
        ''' pred, gt: [B, *] '''
        p, g = pred.flatten(1), gt.flatten(1)
        num = 2 * torch.sum(p * g, dim=1) + eps
        den = torch.sum(p, dim=1) + torch.sum(g, dim=1) + eps
        return 1 - (num/den)
    
    affine_transform_opt = run_affine_transform_3d if is3d else run_affine_transform_2d
    ### Start training
    for epoch in range(start_epoch, cfg.train.epochs):
        # run iterations
        losses_dice = []
        idx = resolve_layer_idx(epoch, cfg)
        print("Using layers upto index %d" % idx)
        for it, batch in enumerate(train_dataloader):
            optim.zero_grad()
            # convert the grayscale images into 3 channel images
            fixed_label, moving_label = batch['source_img'].cuda(), batch['target_img'].cuda()
            if is3d:
                fixed_img = 2*fixed_label - 1
                moving_img = 2*moving_label - 1
            else:
                fixed_img = 2*fixed_label.repeat(1, 3, 1, 1) - 1
                moving_img = 2*moving_label.repeat(1, 3, 1, 1) - 1
            # get features
            if cfg.train.train_all_levels:
                fixed_features_list, moving_features_list = model(fixed_img)[:idx+1], model(moving_img)[:idx+1]
            else:
                fixed_features_list, moving_features_list = model(fixed_img)[idx:idx+1], model(moving_img)[idx:idx+1]
            # get affine maps
            affine_map_list = [None]
            N = len(fixed_features_list)
            for i in range(N):
                affine_map, losses_opt = affine_transform_opt(fixed_features_list[i], moving_features_list[i], \
                                                                init_affine=affine_map_list[-1], iterations=cfg.diffopt.iterations, debug=True)
                affine_map_list.append(affine_map)
            affine_map_list = affine_map_list[1:]
            # compute loss
            loss_total = 0
            for affine_map in affine_map_list:
                affine_grid = F.affine_grid(affine_map, fixed_label.shape, align_corners=True)
                moved_label = F.grid_sample(moving_label, affine_grid, align_corners=True)
                loss = dice_loss_fn(moved_label, fixed_label)
                loss_total += loss
            loss_total /= len(affine_map_list)
            will_overlap = batch['will_overlap'].float().cuda()
            with torch.no_grad():
                if will_overlap.sum() > 0:
                    loss_overlap = ((loss * will_overlap).mean()/will_overlap.mean()).item()
                else:
                    loss_overlap = 1
                if will_overlap.sum() < cfg.batch_size:
                    loss_non_overlap = ((loss * (1-will_overlap)).mean()/(1-will_overlap).mean()).item()
                else:
                    loss_non_overlap = 1
            loss_total = loss_total.mean()
            loss_total.backward()
            loss = loss.mean()
            # apply loss
            # print details (or to log)
            print("Epoch: %d, Iter: %d, Loss: %.4f, Loss_overlap: %.4f, Loss_nonoverlap: %.4f, frac-nonoverlap: %.4f, lr: %.6f" % (epoch, it, loss.item(), \
                                                                loss_overlap, loss_non_overlap, 1 - will_overlap.mean(), scheduler.get_last_lr()[0]))
            optim.step()
            # add losses to compute best model
            losses_dice.append(loss.item())
            # log statistics
            if cfg.deploy:
                ret_dict = {
                    'loss': loss.item(),
                    'epoch': epoch,
                    'feature_idx': idx,
                    'loss_overlap': loss_overlap,
                    'loss_nonoverlap': loss_non_overlap,
                    'frac_nonoverlap': 1 - will_overlap.mean(),
                }
                if it % 50 == 0:
                    # log the optimization table 
                    loss_opt_wandb = wandb.Table(data=list(enumerate(losses_opt)), columns=['iter', 'loss'])
                    ret_dict['loss_opt'] = wandb.plot.line(loss_opt_wandb, "iter", "loss", title="Optimization Loss")
                    # log the feature images
                    if not is3d:
                        fixed_feat_npy = F.interpolate(fixed_features_list[-1], size=(sz, sz))[0].abs().detach().cpu().numpy().mean(0, keepdims=True)
                        moving_feat_npy = F.interpolate(moving_features_list[-1], size=(sz, sz))[0].abs().detach().cpu().numpy().mean(0, keepdims=True)
                        # add max magnitude
                        ret_dict['max_fixed_magnitude'] = fixed_feat_npy.max()
                        ret_dict['max_moving_magnitude'] = moving_feat_npy.max()
                        moving_feat_npy /= moving_feat_npy.max()
                        fixed_feat_npy /= fixed_feat_npy.max()
                        fixed_image_npy = fixed_img[0, 0].detach().cpu().numpy()[..., None]*0.5 + 0.5
                        moving_image_npy = moving_img[0, 0].detach().cpu().numpy()[..., None]*0.5 + 0.5
                        # add to dict
                        ret_dict['fixed_image'] = wandb.Image(fixed_image_npy)
                        ret_dict['moving_image'] = wandb.Image(moving_image_npy)
                        for i in range(fixed_feat_npy.shape[0]):
                            ret_dict[f'fixed_feat_{i}'] = wandb.Image(fixed_feat_npy[i][..., None])
                            ret_dict[f'moving_feat_{i}'] = wandb.Image(moving_feat_npy[i][..., None])
                else:
                    pass
                wandb.log(ret_dict)

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
                }, os.path.join(cfg.save_dir, 'best_dice_loss.pth'))
    # cleanup logging and wandb
    cleanup(cfg, fp)
    
if __name__ == '__main__':
    try:
        main()
    except:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        wandb.finish()
