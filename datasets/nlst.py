'''
author: rohitrango

Dataset for loading the LungCT dataset (from the learn2reg challenge)
has keypoints
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
try:
    from datasets.abdomen import MINDSSC 
except ImportError:
    from abdomen import MINDSSC 
try:
    from solver.diffeo import ALIGN_CORNERS as align_corners
except ImportError:
    align_corners = False

import pandas as pd

from torch import nn
import torch.nn.functional as F

def process_image(image, mask):
    image = np.clip(image, -1200, 400)
    image = (image - image.min()) / (image.max() - image.min())
    image = image * mask
    return image

def to_torch_kps(kps, shape):
    # shape is [X, Y, Z] and kps = [N, 3=xyz]
    kps = kps.astype(np.float32)
    size = np.array(shape, dtype=np.float32)[None]
    kps = (2*kps+1)/size - 1
    return torch.from_numpy(kps).float()

def get_affine_transform(seed=None, angle=np.pi/6, translation=0.1):
    """Sample 3D affine augmentation 

    Returns: 
        3x4 affine matrix
    """
    # Set random seed if provided
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
        
    # Generate random scaling factors between 0.8 and 1.2
    scales = rng.uniform(0.8, 1.2, size=3)
    # Create random rotation angles
    angles = rng.uniform(-angle, angle, size=3)  # ±30 degrees
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
    A = Rx @ Ry @ Rz @ np.diag(scales)
    t = rng.uniform(-translation, translation, size=3)
    # print(A, t, angles * 180 / np.pi)
    affine = np.concatenate([A, t[:, None]], axis=1)
    affine = np.concatenate([affine, np.array([[0, 0, 0, 1]])], axis=0)
    return torch.from_numpy(affine).float()

class NLST(Dataset):
    ''' dataset that returns images and keypoints '''
    def __init__(self, data_root='data/NLST', split='train', use_mind=False, aug=False):
        super().__init__()
        # self.images = sorted(glob(osp.join(data_root, 'imagesTr' if split != 'test' else 'imagesTs', '*.nii.gz')))
        # self.masks = sorted(glob(osp.join(data_root, 'masksTr' if split != 'test' else 'masksTs', '*.nii.gz')))
        # self.keypoints = sorted(glob(osp.join(data_root, 'keypointsTr', '*.csv'))) if split != 'test' else [None for _ in range(len(self.images))]
        with open(osp.join(data_root, 'NLST_dataset.json'), 'r') as f:
            metadata = json.load(f)
        
        if split == 'train':
            self.reg_list = metadata['training_paired_images']
        elif split == 'val':
            self.reg_list = metadata['registration_val']
        else:
            raise ValueError(f"Invalid split: {split}")
        
        for item in self.reg_list:
            item['fixed'] = osp.join(data_root, item['fixed'])
            item['moving'] = osp.join(data_root, item['moving'])
            item['fixed_kps'] = item['fixed'].replace('imagesTr', 'keypointsTr').replace('.nii.gz', '.csv')
            item['moving_kps'] = item['moving'].replace('imagesTr', 'keypointsTr').replace('.nii.gz', '.csv')
            item['fixed_mask'] = item['fixed'].replace('imagesTr', 'masksTr')
            item['moving_mask'] = item['moving'].replace('imagesTr', 'masksTr')

        # if not test, split the images into train and val
        self.use_mind = use_mind
        self.aug = aug

        self.split = split
        self.N = len(self.reg_list)

    def __len__(self):
        return self.N
    
    def __getitem__(self, idx):
        # if idx < self.N:
        item = self.reg_list[idx]
        if self.split == 'train':
            if np.random.rand() < 0.5:
                f_images = item['fixed']
                m_images = item['moving']
                f_keypoints = item['fixed_kps']
                m_keypoints = item['moving_kps']
                f_mask = item['fixed_mask']
                m_mask = item['moving_mask']
            else:
                # flip the images and keypoints
                f_images = item['moving']
                m_images = item['fixed']
                f_keypoints = item['moving_kps']
                m_keypoints = item['fixed_kps']
                f_mask = item['moving_mask']
                m_mask = item['fixed_mask']
        else:
            f_images = item['fixed']
            m_images = item['moving']
            f_keypoints = item['fixed_kps']
            m_keypoints = item['moving_kps']
            f_mask = item['fixed_mask']
            m_mask = item['moving_mask']
        
        subject_name = osp.basename(f_images).split('_')[1]
        spacing = nib.load(m_images).header.get_zooms()[:3]
        # given the images and masks, get the keypoints
        f_mask, m_mask = nib.load(f_mask).get_fdata().squeeze(), nib.load(m_mask).get_fdata().squeeze()
        f_image = process_image(nib.load(f_images).get_fdata().squeeze(), f_mask)
        m_image = process_image(nib.load(m_images).get_fdata().squeeze(), m_mask)
        # use mind features if specified
        if self.use_mind:
            f_image = MINDSSC(torch.from_numpy(f_image)[None, None].float().cuda())[0].cpu().numpy() * f_mask[None]
            m_image = MINDSSC(torch.from_numpy(m_image)[None, None].float().cuda())[0].cpu().numpy() * m_mask[None]
        else:
            f_image = f_image[None]
            m_image = m_image[None]
        
        # only for train and val
        if f_keypoints is not None:
            # load keypoints
            f_kps = np.array(pd.read_csv(f_keypoints)) + 0
            m_kps = np.array(pd.read_csv(m_keypoints)) + 0
            # convert to torch tensors and normalize
            f_kps = to_torch_kps(f_kps, f_image.shape[-3:]) + 0
            m_kps = to_torch_kps(m_kps, m_image.shape[-3:]) + 0
        else:
            f_kps = 0
            m_kps = 0

        # change the image to [ZYX] format from [XYZ]
        f_image = f_image.transpose(0, 3, 2, 1) + 0
        m_image = m_image.transpose(0, 3, 2, 1) + 0

        # transpose masks too, later we flip them and check if keypoints are in the mask
        f_mask = (f_mask[None]).transpose(0, 3, 2, 1) + 0
        m_mask = (m_mask[None]).transpose(0, 3, 2, 1) + 0

        # augment
        # right now images are in [XYZ] format, and so are the keypoints
        p = 1 - 0.5**(1/3)
        if self.split == 'train' and self.aug:
            # random flip 
            if np.random.rand() < p:
                f_image, m_image = f_image[:, ::-1], m_image[:, ::-1]
                f_mask, m_mask = f_mask[:, ::-1], m_mask[:, ::-1]
                f_kps[:, 2], m_kps[:, 2] = -f_kps[:, 2], -m_kps[:, 2]
            if np.random.rand() < p:
                f_image, m_image = f_image[:, :, ::-1], m_image[:, :, ::-1]
                f_mask, m_mask = f_mask[:, :, ::-1], m_mask[:, :, ::-1]
                f_kps[:, 1], m_kps[:, 1] = -f_kps[:, 1], -m_kps[:, 1]
            if np.random.rand() < p:
                f_image, m_image = f_image[:, :, :, ::-1], m_image[:, :, :, ::-1]
                f_mask, m_mask = f_mask[:, :, :, ::-1], m_mask[:, :, :, ::-1]
                f_kps[:, 0], m_kps[:, 0] = -f_kps[:, 0], -m_kps[:, 0]
            f_image = f_image + 0
            m_image = m_image + 0
            f_mask = f_mask + 0
            m_mask = m_mask + 0

            # # random rotation
            if np.random.rand() < 0.5:
                shape = [1, 1] + list(f_image.shape[-3:])
                affine = get_affine_transform()
                affineinv = torch.linalg.inv(affine)
                # affineinv = affine
                # print(affineinv.shape)
                # affineinv = affine
                grid = F.affine_grid(affineinv[:3][None], shape, align_corners=align_corners)
                f_image = F.grid_sample(torch.from_numpy(f_image).float().unsqueeze(0), grid, align_corners=align_corners)[0].numpy()
                m_image = F.grid_sample(torch.from_numpy(m_image).float().unsqueeze(0), grid, align_corners=align_corners)[0].numpy()
                # filter out the values that are not in the mask
                f_mask = F.grid_sample(torch.from_numpy(f_mask).float().unsqueeze(0), grid, align_corners=align_corners)
                m_mask = F.grid_sample(torch.from_numpy(m_mask).float().unsqueeze(0), grid, align_corners=align_corners)
                # filter vals, (f_mask = [1, 1, H, W, D]), f_kps = [K, 3] -> [1, 1, 1, K, 3]
                f_vals = F.grid_sample(f_mask, f_kps[None, None, None], align_corners=align_corners).squeeze()   # [1, 1, 1, 1, K]
                m_vals = F.grid_sample(m_mask, m_kps[None, None, None], align_corners=align_corners).squeeze()
                # filter out the values that are not in the mask
                idx = torch.where((f_vals >= 0.5) & (m_vals >= 0.5))[0]
                f_kps = f_kps[idx]
                m_kps = m_kps[idx]
                # print(affine)
                # keypoints also move
                f_kps = f_kps @ affine[:3, :3].T + affine[:3, 3][None]
                m_kps = m_kps @ affine[:3, :3].T + affine[:3, 3][None]

        ret = {
            'source_img': torch.tensor(f_image, dtype=torch.float32),
            'target_img': torch.tensor(m_image, dtype=torch.float32),
            'source_kps': f_kps,
            'target_kps': m_kps,
            'spacing': torch.tensor(spacing, dtype=torch.float32),
            'subject_name': subject_name,
        }
        return ret
