'''
author: rohitrango

Dataset for loading the OASIS dataset (from the learn2reg challenge)
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

class AbdomenMRCT(Dataset):
    def __init__(self, data_root, split='train', affine_augmentation_prob=0.2) -> None:
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.affine_augmentation_prob = affine_augmentation_prob

        images = sorted(glob(osp.join(data_root, 'imagesTr', '*.nii.gz')))
        images0, images1 = separate_modalities(images)
        labels = sorted(glob(osp.join(data_root, 'labelsTr', '*.nii.gz')))
        labels0, labels1 = separate_modalities(labels)

        # load metadata
        with open(osp.join(data_root, 'AbdomenMRCT_dataset.json'), 'r') as f:
            metadata = json.load(f)

        if split == 'train':
            self.images = images0[:-10] + images1[:-10]
            self.labels = labels0[:-10] + labels1[:-10]
            self.pairs = list(itertools.combinations(range(len(self.images)), 2))
            self.pairs = self.pairs + [(y, x) for x, y in self.pairs]

        elif split == 'val':
            self.images = images0[-10:] + images1[-10:]
            self.labels = labels0[-10:] + labels1[-10:]
            self.pairs = list(itertools.combinations(range(len(self.images)), 2))
            self.pairs = self.pairs + [(y, x) for x, y in self.pairs]

        elif split == 'test':
            pairs = []
            images = []
            i = 0
            for item in metadata['registration_test']:
                fixed, moving = item['fixed'], item['moving']
                fixed, moving = osp.join(data_root, fixed), osp.join(data_root, moving)
                images.append(fixed)
                images.append(moving)
                pairs.append((i, i+1))
                i += 2
            self.images = images
            self.labels = []
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
        center = np.array(image.shape) // 2
        offset = center - affine_matrix @ center
        
        aug_image = affine_transform(image, affine_matrix, offset=offset, order=1)
        
        if label is not None:
            # Convert label to numpy, apply transform, and convert back
            aug_label = affine_transform(label, affine_matrix, offset=offset, order=0)
            return aug_image, aug_label
        
        return aug_image

    def __getitem__(self, index):
        fixed, moving = [self.images[x] for x in self.pairs[index]]
        if self.labels != []:
            fixedlab, movlab = [self.labels[x] for x in self.pairs[index]]
        else:
            fixedlab, movlab = None, None
        
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
        
        if self.split == 'test':
            return {
                'source_img': torch.from_numpy(fixedimg).unsqueeze(0).float(),
                'target_img': torch.from_numpy(movingimg).unsqueeze(0).float(),
                'source_img_path': fixed,
                'target_img_path': moving
            }
        
        # for train and val, we load the label, and augment if the split is train
        fixedlabdata = nib.load(fixedlab).get_fdata().squeeze().astype(np.int32)
        movinglabdata = nib.load(movlab).get_fdata().squeeze().astype(np.int32)

        # augment
        if self.split == 'train':
            for ax in range(3):
                if np.random.rand() < 0.5:
                    fixedimg = np.flip(fixedimg, axis=ax)
                    fixedlabdata = np.flip(fixedlabdata, axis=ax)
                    movingimg = np.flip(movingimg, axis=ax)
                    movinglabdata = np.flip(movinglabdata, axis=ax)
            
            # Apply 3D affine augmentation with same seed for fixed and moving images
            if np.random.rand() < self.affine_augmentation_prob:  # 20% chance to apply augmentation
                seed = np.random.randint(0, 2**32)
                fixedimg, fixedlabdata = self.apply_3d_augmentation(fixedimg, fixedlabdata, seed=seed)
                movingimg, movinglabdata = self.apply_3d_augmentation(movingimg, movinglabdata, seed=seed)
        
        return {
            'source_img': torch.from_numpy(fixedimg+0).unsqueeze(0).float(),
            'target_img': torch.from_numpy(movingimg+0).unsqueeze(0).float(),
            'source_img_path': fixed,
            'target_img_path': moving,
            'source_label': torch.from_numpy(fixedlabdata+0).unsqueeze(0).long(),
            'target_label': torch.from_numpy(movinglabdata+0).unsqueeze(0).long(),
            'source_label_path': fixedlab,
            'target_label_path': movlab
        }

if __name__ == '__main__':
    pass