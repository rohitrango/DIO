''' this model is to reconstruct the same image from a single level of the TransFeX model'''
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 
# make sure the parent of this folder is in path to be 
# able to access everything
from models.TransMorph import TransFeX
from models.configs_TransMorph import get_3DTransFeX_config
from losses import NCC_vxm, DiceLossWithLongLabels 
from solver.adam import multi_scale_diffeomorphic_solver
from solver.utils import gaussian_1d, img2v_3d, v2img_3d
# logging
import wandb
import hydra
from model_utils import displacements_to_warps
from utils import set_seed, init_wandb, open_log, cleanup
from datasets.oasis import OASISImageOnly
import numpy as np

datasets = {
    'oasis': OASISImageOnly
}

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
    model_cfg = get_3DTransFeX_config()
    model_cfg['levels'] = [cfg.model.levels]
    model_cfg['output_channels'] = cfg.model.output_channels
    # change any model config here
    # load model and optionally weights
    model = TransFeX(model_cfg).cuda()
    optim = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    scheduler = torch.optim.lr_scheduler.PolynomialLR(optim, cfg.train.epochs, power=cfg.train.lr_power_decay)
    start_epoch = 0
    best_image_loss = np.inf
    if cfg.model_path is not None:
        saved_data = torch.load(cfg.model_path)
        model.load_state_dict(saved_data['model'])
        optim.load_state_dict(saved_data['optim'])
        scheduler.load_state_dict(saved_data['scheduler'])
        print(f"Loaded model and optim from: {cfg.model_path} at epoch: {saved_data['epoch']}.")
        # load other metrics
        start_epoch = saved_data['epoch'] + 1
        best_image_loss = saved_data['best_image_loss']
    model.train()
    
    # Just train a reconstruction model
    for epoch in range(start_epoch, cfg.train.epochs):
        # run iterations
        losses_mse = []
        for it, batch in enumerate(train_dataloader):
            optim.zero_grad()
            source_img, transformed_img = batch['source_img'].cuda(), batch['transformed_img'].cuda()
            # get features
            features = model(transformed_img)[0]
            features = features.mean(1, keepdim=True)
            size = features.shape[2:]
            true_img = F.interpolate(source_img, size=size, mode='trilinear', align_corners=True)
            # get loss
            loss = F.mse_loss(true_img, features)
            loss.backward()
            # apply loss
            # print details (or to log)
            print("Epoch: %d, Iter: %d, MSE: %.4f, lr: %.4f" % (epoch, it, loss.item(), scheduler.get_last_lr()[0]))
            optim.step()
            # add losses to compute best model
            losses_mse.append(loss.item())
            # log statistics
            if cfg.deploy:
                if it % 50 == 0:
                    # make true and predicted images into wandb images
                    D = int(true_img.shape[2] // 2)
                    true_img = np.clip(true_img.detach().cpu().numpy()[0, 0, D] * 0.5 + 0.5, 0, 1) * 255
                    true_img = np.repeat(true_img.astype(np.uint8)[..., None], 3, axis=-1)
                    # predicted
                    predicted_img = np.clip(features.detach().cpu().numpy()[0, 0, D] * 0.5 + 0.5, 0, 1) * 255
                    predicted_img = np.repeat(predicted_img.astype(np.uint8)[..., None], 3, axis=-1)
                    wandb.log({
                        'loss_mse': loss.item(),
                        'true_img': wandb.Image(true_img, caption='True Image'),
                        'predicted_img': wandb.Image(predicted_img, caption='Predicted Image'),
                    })
                else:
                    wandb.log({
                        'loss_mse': loss.item(),
                    })
        # scheduler step
        scheduler.step()
        if cfg.deploy:
            # save best model according to dice loss
            if np.mean(losses_mse) < best_image_loss:
                best_image_loss = np.mean(losses_mse)
                torch.save({
                    'model': model.state_dict(),
                    'optim': optim.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch,
                    'best_image_loss': np.mean(losses_mse)
                }, os.path.join(cfg.save_dir, 'best_model.pth'))
    # cleanup logging and wandb
    cleanup(cfg, fp)
    

if __name__ == '__main__':
    main()
