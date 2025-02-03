'''
author: rohitrango

Dataset for loading the AbdomenCT dataset (from the learn2reg challenge)
'''
import torch
import nibabel as nib
from torch.utils.data import Dataset
from glob import glob
import os
import os.path as osp
import numpy as np
import pprint
from PIL import Image
import json
import itertools
from scipy.ndimage import affine_transform

from torch import nn
import torch.nn.functional as F

def get_multimodal_pairs(images):
    ''' get pairs of images that are of different modalities '''
    pairs = []
    for i in range(len(images)):
        for j in range(i+1, len(images)):
            is_ct_i = is_ct_image(images[i])
            is_ct_j = is_ct_image(images[j])
            if is_ct_i != is_ct_j:
                pairs.append((i, j))
                pairs.append((j, i))
    return pairs

def is_ct_image(image):
    return '0001.nii.gz' in image

def separate_modalities(images):
    images0 = []
    images1 = []
    for image in images:
        if '0000.nii.gz' in image:
            images0.append(image)
        else:
            images1.append(image)
    return images0, images1

def preprocess_ct(image):
    image = np.clip(image, -300, 500)
    minval, maxval = -300, 500
    image = (image - minval) / (maxval - minval)
    return image

def preprocess_mr(image, percentile=95):
    percentile = np.percentile(image, percentile)
    image = np.clip(image, 0, percentile)
    image = image / percentile
    return image

def pdist_squared(x):
    xx = (x**2).sum(dim=1).unsqueeze(2)
    yy = xx.permute(0, 2, 1)
    dist = xx + yy - 2.0 * torch.bmm(x.permute(0, 2, 1), x)
    dist[dist != dist] = 0
    dist = torch.clamp(dist, 0.0, np.inf)
    return dist

def MINDSSC(img, radius=2, dilation=2, device='cuda'):
    # see http://mpheinrich.de/pub/miccai2013_943_mheinrich.pdf for details on the MIND-SSC descriptor
    # code taken from: https://github.com/multimodallearning/convexAdam/blob/main/src/convexAdam/convex_adam_utils.py
    
    # kernel size
    kernel_size = radius * 2 + 1
    
    # define start and end locations for self-similarity pattern
    six_neighbourhood = torch.Tensor([[0,1,1],
                                      [1,1,0],
                                      [1,0,1],
                                      [1,1,2],
                                      [2,1,1],
                                      [1,2,1]]).long()
    
    # squared distances
    dist = pdist_squared(six_neighbourhood.t().unsqueeze(0)).squeeze(0)
    
    # define comparison mask
    x, y = torch.meshgrid(torch.arange(6), torch.arange(6), indexing='ij')
    mask = ((x > y).view(-1) & (dist == 2).view(-1))
    
    # build kernel
    idx_shift1 = six_neighbourhood.unsqueeze(1).repeat(1,6,1).view(-1,3)[mask,:]
    idx_shift2 = six_neighbourhood.unsqueeze(0).repeat(6,1,1).view(-1,3)[mask,:]
    mshift1 = torch.zeros(12, 1, 3, 3, 3).to(device)
    mshift1.view(-1)[torch.arange(12) * 27 + idx_shift1[:,0] * 9 + idx_shift1[:, 1] * 3 + idx_shift1[:, 2]] = 1
    mshift2 = torch.zeros(12, 1, 3, 3, 3).to(device)
    mshift2.view(-1)[torch.arange(12) * 27 + idx_shift2[:,0] * 9 + idx_shift2[:, 1] * 3 + idx_shift2[:, 2]] = 1
    rpad1 = nn.ReplicationPad3d(dilation)
    rpad2 = nn.ReplicationPad3d(radius)
    # compute patch-ssd
    ssd = F.avg_pool3d(rpad2((F.conv3d(rpad1(img), mshift1, dilation=dilation) - F.conv3d(rpad1(img), mshift2, dilation=dilation)) ** 2), kernel_size, stride=1)
    # MIND equation
    mind = ssd - torch.min(ssd, 1, keepdim=True)[0]
    mind_var = torch.mean(mind, 1, keepdim=True)
    mind_var = torch.clamp(mind_var, mind_var.mean().item()*0.001, mind_var.mean().item()*1000)
    mind /= mind_var
    mind = torch.exp(-mind)
    #permute to have same ordering as C++ code
    mind = mind[:, torch.Tensor([6, 8, 1, 11, 2, 10, 0, 7, 9, 4, 5, 3]).long(), :, :, :]
    return mind


class AbdomenMRCT(Dataset):
    def __init__(self, data_root, split='train', affine_augmentation_prob=0.2) -> None:
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.affine_augmentation_prob = affine_augmentation_prob

        images = sorted(glob(osp.join(data_root, 'imagesTr', '*.nii.gz')))
        images0, images1 = separate_modalities(images)
        masks = sorted(glob(osp.join(data_root, 'masksTr', '*.nii.gz')))
        masks0, masks1 = separate_modalities(masks)
        labels = sorted(glob(osp.join(data_root, 'labelsTr', '*.nii.gz')))
        labels0, labels1 = separate_modalities(labels)

        # load metadata
        with open(osp.join(data_root, 'AbdomenMRCT_dataset.json'), 'r') as f:
            metadata = json.load(f)

        N = 5
        if split == 'train':
            self.images = images0[:-N] + images1[:-N]
            self.labels = labels0[:-N] + labels1[:-N]
            self.masks = masks0[:-N] + masks1[:-N]
            self.pairs = get_multimodal_pairs(self.images)

        elif split == 'val':
            self.images = images0[-N:] + images1[-N:]
            self.labels = labels0[-N:] + labels1[-N:]
            self.masks = masks0[-N:] + masks1[-N:]
            self.pairs = get_multimodal_pairs(self.images)

        elif split == 'test':
            pairs = []
            images = []
            masks = []
            i = 0
            for item in metadata['registration_test']:
                fixed, moving = item['fixed'], item['moving']
                fixed, moving = osp.join(data_root, fixed), osp.join(data_root, moving)
                images.append(fixed)
                images.append(moving)
                masks.append(fixed.replace("images", "masks"))
                masks.append(moving.replace("images", "masks"))
                pairs.append((i, i+1))
                i += 2
            self.images = images
            self.labels = []
            self.masks = []
            self.pairs = pairs
        
        # max label index
        self.max_label_index = 4
    
    def __len__(self):
        return len(self.pairs)
    
    @staticmethod
    def apply_3d_augmentation(image, label=None, seed=None):
        """Apply 3D affine augmentation to image and optionally label data.
        
        Args:
            image: numpy array of image data
            label: optional tensor of label data
            seed: optional random seed for reproducible transformations
            
        Returns:
            Tuple of (augmented_image, augmented_label) if label is provided,
            otherwise returns just augmented_image
        """
        # Set random seed if provided
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random.RandomState()
            
        # Generate random scaling factors between 0.8 and 1.2
        scales = rng.uniform(0.8, 1.2, size=3)
        
        # Create random rotation angles
        angles = rng.uniform(-np.pi/6, np.pi/6, size=3)  # ±30 degrees
        
        # Create rotation matrices for each axis
        Rx = np.array([[1, 0, 0],
                      [0, np.cos(angles[0]), -np.sin(angles[0])],
                      [0, np.sin(angles[0]), np.cos(angles[0])]])
        
        Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                      [0, 1, 0],
                      [-np.sin(angles[1]), 0, np.cos(angles[1])]])
        
        Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                      [np.sin(angles[2]), np.cos(angles[2]), 0],
                      [0, 0, 1]])
        
        # Combine rotations and scaling
        affine_matrix = Rx @ Ry @ Rz @ np.diag(scales)
        
        # Apply transformation using scipy's affine_transform
        center = np.array(image.shape[-3:]) // 2
        offset = center - affine_matrix @ center
        
        C = image.shape[0]
        aug_image = [0] * C
        for i in range(C):
            aug_image[i] = affine_transform(image[i], affine_matrix, offset=offset, order=1)
        aug_image = np.stack(aug_image, axis=0)
        
        if label is not None:
            # Convert label to numpy, apply transform, and convert back
            aug_label = affine_transform(label, affine_matrix, offset=offset, order=0)
            return aug_image, aug_label
        
        return aug_image

    def __getitem__(self, index):
        fixed, moving = [self.images[x] for x in self.pairs[index]]
        fixedmask, movingmask = [self.masks[x] for x in self.pairs[index]]
        if self.labels != []:
            fixedlab, movlab = [self.labels[x] for x in self.pairs[index]]
        else:
            fixedlab, movlab = None, None
        
        fixedmask = nib.load(fixedmask).get_fdata().squeeze()
        movingmask = nib.load(movingmask).get_fdata().squeeze()
        # load image
        fixedimg = nib.load(fixed).get_fdata().squeeze()
        movingimg = nib.load(moving).get_fdata().squeeze()
        if is_ct_image(fixed):
            fixedimg = preprocess_ct(fixedimg)
        else:
            fixedimg = preprocess_mr(fixedimg)
        
        if is_ct_image(moving):
            movingimg = preprocess_ct(movingimg)
        else:
            movingimg = preprocess_mr(movingimg)

        # apply mask to image
        fixedimg = fixedimg * fixedmask
        movingimg = movingimg * movingmask

        # compute MIND-SSC descriptor
        fixedimg = MINDSSC(torch.from_numpy(fixedimg)[None, None].float().cuda())[0].cpu().numpy() * fixedmask[None]
        movingimg = MINDSSC(torch.from_numpy(movingimg)[None, None].float().cuda())[0].cpu().numpy() * movingmask[None]
        # fixedimg = 2*fixedimg - 1
        # movingimg = 2*movingimg - 1

        if self.split == 'test':
            return {
                'source_img': torch.from_numpy(fixedimg).float(),
                'target_img': torch.from_numpy(movingimg).float(),
                'source_img_path': fixed,
                'target_img_path': moving
            }
        
        # for train and val, we load the label, and augment if the split is train
        fixedlabdata = nib.load(fixedlab).get_fdata().squeeze().astype(np.int32)
        movinglabdata = nib.load(movlab).get_fdata().squeeze().astype(np.int32)

        # augment
        if self.split == 'train':
            # for ax in range(3):
            #     if np.random.rand() < 0.5:
            #         C = fixedimg.shape[0]
            #         for i in range(C):
            #             fixedimg[i] = np.flip(fixedimg[i], axis=ax)
            #             movingimg[i] = np.flip(movingimg[i], axis=ax)
            #         # also flip the label
            #         fixedlabdata = np.flip(fixedlabdata, axis=ax)
            #         movinglabdata = np.flip(movinglabdata, axis=ax)
            
            # Apply 3D affine augmentation with same seed for fixed and moving images
            print(fixedimg.shape, movingimg.shape, fixedlabdata.shape, movinglabdata.shape)
            if np.random.rand() < self.affine_augmentation_prob:  # 20% chance to apply augmentation
                seed = np.random.randint(0, 2**32)
                fixedimg, fixedlabdata = self.apply_3d_augmentation(fixedimg, fixedlabdata, seed=seed)
                movingimg, movinglabdata = self.apply_3d_augmentation(movingimg, movinglabdata, seed=seed)
        
        return {
            'source_img': torch.from_numpy(fixedimg+0).float(),
            'target_img': torch.from_numpy(movingimg+0).float(),
            'source_img_path': fixed,
            'target_img_path': moving,
            'source_label': torch.from_numpy(fixedlabdata+0).unsqueeze(0).long(),
            'target_label': torch.from_numpy(movinglabdata+0).unsqueeze(0).long(),
            'source_label_path': fixedlab,
            'target_label_path': movlab
        }

if __name__ == '__main__':
    pass