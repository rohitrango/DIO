''' Dataset of toy examples 
Intended to check measure of how well the DEQ optimization works
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

class SquareDataset(Dataset):
    ''' Dataset that generates random squares that either do or do not overlap entirely '''
    def __init__(self, split='train', N=1000, img_size=256, square_size=32, overlap=0.5, seed=49523):
        super().__init__()
        self.split = split
        self.N = N
        self.img_size = img_size//2   # we will use a smaller image size to make the problem easier (and pad it with the big image_size)
        self.img_size_actual = img_size
        self.square_size = square_size
        self.overlap = overlap
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        # extra convenience variables
        self.ssize2, self.r_ssize2 = self.square_size//2, self.square_size - self.square_size//2

    def __len__(self):
        # just a dummy length to make one epoch meaningful
        return self.N
    
    def __getitem__(self, idx):
        # if train, we want to keep generating new examples
        # if test, we want to fix the generation process, so we start with a new randomstate
        rng = self.rng if self.split == 'train' else np.random.RandomState(self.seed + idx*10)
        will_overlap = rng.rand() < self.overlap
        ssize2, r_ssize2 = self.ssize2, self.r_ssize2
        # f stands for fixed image
        fx, fy = rng.randint(0, self.img_size - self.square_size, size=2)
        fcx, fcy = fx + ssize2, fy + ssize2
        if will_overlap:
            # choose a random center in the overlap region
            mcx = rng.randint(max(ssize2, fcx - ssize2), min(self.img_size - r_ssize2, fcx + ssize2))
            mcy = rng.randint(max(ssize2, fcy - ssize2), min(self.img_size - r_ssize2, fcy + ssize2))
        else:
            # keep sampling until you hit a non-overlapping square
            while True:
                mcx, mcy = ssize2 + rng.randint(0, self.img_size - self.square_size, size=2)
                if np.abs(mcx - fcx) > self.square_size or np.abs(mcy - fcy) > self.square_size:
                    break
        # make the images
        fixed_img = np.zeros((self.img_size, self.img_size))
        fixed_img[fx:fx+self.square_size, fy:fy+self.square_size] = 1
        moving_img = np.zeros((self.img_size, self.img_size))
        moving_img[mcx-ssize2:mcx+r_ssize2, mcy-ssize2:mcy+r_ssize2] = 1
        # pad with zeros
        fixed_img = np.pad(fixed_img, ((self.img_size_actual - self.img_size)//2, (self.img_size_actual - self.img_size)//2), mode='constant')
        moving_img = np.pad(moving_img, ((self.img_size_actual - self.img_size)//2, (self.img_size_actual - self.img_size)//2), mode='constant')
        # output
        ret = {
            'source_img': torch.from_numpy(fixed_img).float()[None],
            'target_img': torch.from_numpy(moving_img).float()[None],
            'source_center': [fcx, fcy],
            'target_center': [mcx, mcy],
            'will_overlap': will_overlap,
        }
        return ret


class Square3DDataset(Dataset):
    ''' Dataset for 3D images that generates random squares that either do or do not overlap entirely '''
    def __init__(self, split='train', N=1000, img_size=128, square_size=20, overlap=0.5, seed=49523):
        super().__init__()
        self.split = split
        self.N = N
        self.img_size = img_size//2   # we will use a smaller image size to make the problem easier (and pad it with the big image_size)
        self.img_size_actual = img_size
        self.square_size = square_size
        self.overlap = overlap
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        # extra convenience variables
        self.ssize2, self.r_ssize2 = self.square_size//2, self.square_size - self.square_size//2

    def __len__(self):
        # just a dummy length to make one epoch meaningful
        return self.N
    
    def __getitem__(self, idx):
        # if train, we want to keep generating new examples
        # if test, we want to fix the generation process, so we start with a new randomstate
        rng = self.rng if self.split == 'train' else np.random.RandomState(self.seed + idx*10)
        will_overlap = rng.rand() < self.overlap
        ssize2, r_ssize2 = self.ssize2, self.r_ssize2
        # f stands for fixed image
        fx, fy, fz = rng.randint(0, self.img_size - self.square_size, size=3)
        fcx, fcy, fcz = fx + ssize2, fy + ssize2, fz + ssize2
        if will_overlap:
            # choose a random center in the overlap region
            mcx = rng.randint(max(ssize2, fcx - ssize2), min(self.img_size - r_ssize2, fcx + ssize2))
            mcy = rng.randint(max(ssize2, fcy - ssize2), min(self.img_size - r_ssize2, fcy + ssize2))
            mcz = rng.randint(max(ssize2, fcz - ssize2), min(self.img_size - r_ssize2, fcz + ssize2))
        else:
            # keep sampling until you hit a non-overlapping square
            while True:
                mcx, mcy, mcz = ssize2 + rng.randint(0, self.img_size - self.square_size, size=3)
                if np.abs(mcx - fcx) > self.square_size or np.abs(mcy - fcy) > self.square_size or np.abs(mcz - fcz) > self.square_size:
                    break
        # make the images
        fixed_img = np.zeros((self.img_size, self.img_size, self.img_size))
        fixed_img[fx:fx+self.square_size, fy:fy+self.square_size, fz:fz+self.square_size] = 1
        moving_img = np.zeros((self.img_size, self.img_size, self.img_size))
        moving_img[mcx-ssize2:mcx+r_ssize2, mcy-ssize2:mcy+r_ssize2, mcz-ssize2:mcz+r_ssize2] = 1
        # pad with zeros
        res = (self.img_size_actual - self.img_size)//2
        res = (res, res)
        fixed_img = np.pad(fixed_img, (res, res, res), mode='constant')
        moving_img = np.pad(moving_img, (res, res, res), mode='constant')
        # output
        ret = {
            'source_img': torch.from_numpy(fixed_img).float()[None],
            'target_img': torch.from_numpy(moving_img).float()[None],
            'source_center': [fcx, fcy, fcz],
            'target_center': [mcx, mcy, mcz],
            'will_overlap': will_overlap,
        }
        return ret
