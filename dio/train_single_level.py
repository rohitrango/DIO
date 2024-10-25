import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 
# make sure the parent of this folder is in path to be 
# able to access everything
from models.TransMorph import TransFeX
from models.unet3d import UNet3D
from models.configs_TransMorph import get_3DTransFeX_config
from solver.adam import multi_scale_diffeomorphic_solver, multi_scale_warp_solver
from solver.utils import gaussian_1d, img2v_3d, v2img_3d
from solver.losses import NCC_vxm, DiceLossWithLongLabels, _get_loss_function_factory
from solver.losses import LocalNormalizedCrossCorrelationLoss
from solver.adam import align_corners

# logging
import wandb
import hydra
from model_utils import displacements_to_warps, downsample
from utils import set_seed, init_wandb, open_log, cleanup
from datasets.oasis import OASIS
import numpy as np

datasets = {
    'oasis': OASIS
}

def torch2wandbimg(tensor):
    tensor_npy = tensor.detach().cpu().numpy()
    tensor_npy = (tensor_npy - tensor_npy.min()) / (tensor_npy.max() - tensor_npy.min())
    return wandb.Image(tensor_npy)

@hydra.main(config_path='./configs', config_name='default')
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
    # load model
    # model_cfg = get_3DTransFeX_config()
    # model_cfg['levels'] = [cfg.model.levels]
    # model_cfg['output_channels'] = cfg.model.output_channels
    # change any model config here
    # load model and optionally weights
    # model = TransFeX(model_cfg).cuda()

    ### Try UNet
    model = UNet3D(1, cfg.model.output_channels, levels=[cfg.model.levels]).cuda()

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
    elif cfg.diffopt.warp_type == 'freeform':
        diffopt_solver = multi_scale_warp_solver
    else:
        raise ValueError(f"Unknown solver: {cfg.diffopt.solver}")
    
    model.train()
    # load NCC and Dice losses here
    ncc_loss_fn = _get_loss_function_factory(cfg.diffopt.feature_loss_fn, cfg)
    ncc_img_loss_fn = LocalNormalizedCrossCorrelationLoss(kernel_size=7, reduction='mean')
    # ncc_img_loss_fn = NCC_vxm(win=7, inp_channels=1)
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
            fixed_img, moving_img = batch['source_img'].cuda(), batch['target_img'].cuda()
            fixed_label, moving_label = batch['source_label'].cuda(), batch['target_label'].cuda()
            # get features
            if cfg.train.backprop_both:
                fixed_features, moving_features = model(fixed_img), model(moving_img)
            else:
                U = np.random.rand() < 0.5
                with torch.set_grad_enabled(U):
                    fixed_features = model(fixed_img)
                with torch.set_grad_enabled(not U):
                    moving_features = model(moving_img)
            
            # print(f"Fixed features: {fixed_features[0].shape}, Moving features: {moving_features[0].shape}")
            # now get displacements (minimum of 10 iterations)
            iterations = min((epoch + 1)*10, cfg.diffopt.iterations) if cfg.diffopt.gradual else cfg.diffopt.iterations
            displacements, losses_opt = diffopt_solver(fixed_features, moving_features,
                                iterations=[iterations], loss_function=ncc_loss_fn, 
                                debug=True,
                                gaussian_grad=gaussian_grad, gaussian_warp=gaussian_warp)
            losses_opt = losses_opt[0]
            warp = displacements_to_warps(displacements)[0] # only one level
            warp.retain_grad()
            # if cfg.model.levels != 1:
            #     warp = img2v_3d(F.upsample(v2img_3d(warp), size=fixed_img.shape[2:], mode='trilinear', align_corners=align_corners))
            with torch.no_grad():
                if cfg.model.levels != 1:
                    N = cfg.model.levels
                    fixed_img_down = downsample(fixed_img, size=fixed_features[0].shape[2:], mode='trilinear')
                    fixed_label_down = F.interpolate(fixed_label.float(), size=fixed_features[0].shape[2:], mode='nearest').long()
                else:
                    fixed_img_down = fixed_img
                    fixed_label_down = fixed_label
                
            # apply loss
            moved_img = F.grid_sample(moving_img, warp, align_corners=align_corners)
            loss_ncc = 1+ncc_img_loss_fn(moved_img, fixed_img_down)
            loss_dice = dice_loss_fn(moving_label, fixed_label_down, warp)
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
                # loss_mse = 0.5 * (F.mse_loss(moving_features.mean(1, keepdim=True), moving_img) + F.mse_loss(fixed_features.mean(1, keepdim=True), fixed_img))
                loss_mse = F.mse_loss(F.interpolate(model.decode_features(moving_features)[0], size=moving_img.shape[2:], mode='trilinear', align_corners=align_corners), moving_img) \
                    + F.mse_loss(F.interpolate(model.decode_features(fixed_features)[0], size=fixed_img.shape[2:], mode='trilinear', align_corners=align_corners), fixed_img)
                # loss_mse = F.mse_loss(F.interpolate(moving_features[0].mean(1, keepdim=True), size=moving_img.shape[2:], mode='trilinear', align_corners=align_corners), moving_img)
                # loss_mse = loss_mse + F.mse_loss(F.interpolate(fixed_features[0].mean(1, keepdim=True), size=fixed_img.shape[2:], mode='trilinear', align_corners=align_corners), fixed_img)
                loss = loss + (cfg.loss.decay_mse**epoch) * 0.5 * loss_mse
            loss.backward()
            # print details (or to log)
            print("Epoch: %d, Iter: %d, NCC: %.4f, Diceloss: %.4f, MSE: %.4f, mse_lambda: %.4f, lr: %.4f, diffopt_iters: %d" % (epoch, it, loss_ncc.item(), mean_loss_dice.item(), \
                                                            loss_mse.item(), (cfg.loss.decay_mse**epoch) , scheduler.get_last_lr()[0], iterations))
            optim.step()
            if not cfg.deploy:
                if it == 50:
                    break
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
                if it % 50 == 0:
                    B, _, Hi, Wi, Di = fixed_img.shape
                    B, C, Hf, Wf, Df = fixed_features[0].shape
                    with torch.no_grad():
                        fixed_img_sample_idx = np.random.randint(0, 10)-5 + Hi//2
                        fixed_img_sample, moving_img_sample = fixed_img[0, 0, fixed_img_sample_idx], moving_img[0, 0, fixed_img_sample_idx]
                        moved_img_sample = moved_img[0, 0, fixed_img_sample_idx//cfg.model.levels]
                        fixed_feature_sample_idx = np.random.randint(0, 10)-5 + Hf//2
                        # fixed_img_sample, moving_img_sample = fixed_img[0, 0, fixed_img_sample_idx], moving_img[0, 0, fixed_img_sample_idx]
                        log_dict['fixed_img'] = torch2wandbimg(fixed_img_sample)
                        log_dict['moving_img'] = torch2wandbimg(moving_img_sample)
                        log_dict['moved_img'] = torch2wandbimg(moved_img_sample)
                        for c in range(C):
                            fixed_feature_sample, moving_feature_sample = fixed_features[0][0, c, fixed_feature_sample_idx], moving_features[0][0, c, fixed_feature_sample_idx]
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
