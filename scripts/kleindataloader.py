# this is a generic dataloader for Klein et al datasets
import os
from os import path as osp
from torch.utils.data import Dataset
from glob import glob
from natsort import natsorted
import itertools
import nibabel as nib
from scipy.ndimage import zoom
import numpy as np
import torch

def load_pair(image, label, isotropic, crop):
    # load image
    nii = nib.load(image)
    img = nii.get_fdata().squeeze()
    scale = nii.header.get_zooms()[:3]
    # print(img.shape, nii.get_fdata().shape, scale)
    if isotropic:
        img = zoom(img, scale, order=1)
    # img = img.transpose(0, 2, 1)[::-1, ::-1]
    # load label
    nii_l = nib.load(label)
    label = nii_l.get_fdata().squeeze()
    scale = nii_l.header.get_zooms()[:3]
    if isotropic:
        label = zoom(label, scale, order=0)
    # label = label.transpose(0, 2, 1)[::-1, ::-1]
    img = img / img.max() 
    img[img < 0] = 0
    # crop this 
    if crop:
        H, W, D = img.shape
        # extra padding to add
        padding = [max(0, 160-H), max(0, 224-W), max(0, 192-D)]
        padding = [(x//2, x - x//2) for x in padding]
        img = np.pad(img, padding, mode='constant')
        label = np.pad(label, padding, mode='constant')
        # center crop
        H, W, D = img.shape
        img = img[H//2-80:H//2+80, W//2-112:W//2+112, D//2-96:D//2+96]
        label = label[H//2-80:H//2+80, W//2-112:W//2+112, D//2-96:D//2+96]
    else:
        # crop is false, pad it with zeros to make it multiple of 16
        H, W, D = img.shape
        div = 16
        rem = [H % div, W % div, D % div]
        rem = [0 if x == 0 else div - x for x in rem]
        pad = [(x//2, x - x//2) for x in rem]
        img = np.pad(img, pad, mode='constant')
        label = np.pad(label, pad, mode='constant')

    return img, label

class KleinDatasets(Dataset):
    def __init__(self, data_root="/data/rohitrango/brain_data/", dataset="IBSR18", isotropic=True, crop=False, dry_run=False, dry_run_size=5):
        super().__init__()
        assert dataset in ['MGH10', 'CUMC12', 'IBSR18', 'LPBA40']
        self.data_root = osp.join(data_root, dataset)
        # get pairs ready
        if dataset == 'MGH10':
            brains = natsorted(glob(osp.join(self.data_root, 'Brains', 'g*.img')))
            labels = natsorted(glob(osp.join(self.data_root, 'AtlasesCommonLabels', 'g*.img')))
        elif dataset == 'CUMC12':
            brains = natsorted(glob(osp.join(self.data_root, 'Brains', 'm*.img')))
            labels = natsorted(glob(osp.join(self.data_root, 'AtlasesCommonLabels', 'm*.img')))
        elif dataset == 'IBSR18':
            brains = natsorted(glob(osp.join(self.data_root, 'IBSR*', 'IBSR_*_ana_strip.nii.gz')))
            labels = natsorted(glob(osp.join(self.data_root, 'IBSR*', 'IBSR_*_seg_ana_common.nii.gz')))
        # create all pairings if not LPBA
        if dataset != 'LPBA40':
            N = len(brains)
            pairs = list(itertools.product(range(N), range(N)))
            pairs = list(filter(lambda x: x[0] != x[1], pairs))
            self.pairs = [(brains[i], labels[i], brains[j], labels[j]) for i, j in pairs]
            self.pair_ids = pairs
        else:
            # LPBA has all pairs already
            N = 40
            pairs = list(itertools.product(range(1, N+1), range(1, N+1)))
            pairs = list(filter(lambda x: x[0] != x[1], pairs))
            self.pair_ids = pairs
            self.pairs = []
            # i is moving
            for i, j in pairs:
                moving = osp.join(self.data_root, 'registered_pairs', f'l{i}_to_l{j}.img')
                fixed = osp.join(self.data_root, 'registered_pairs', f'l{j}_to_l{j}.img')
                moving_seg = osp.join(self.data_root, 'registered_label_pairs_common', f'l{i}_to_l{j}.img')
                fixed_seg = osp.join(self.data_root, 'registered_label_pairs_common', f'l{j}_to_l{j}.img')
                self.pairs.append((moving, moving_seg, fixed, fixed_seg))
        # dry run
        if dry_run:
            rng = np.random.RandomState(563241)
            rng.shuffle(self.pairs)
            self.pairs = self.pairs[:dry_run_size]
        self.isotropic = isotropic
        self.crop = crop
        # get maxlabel index
        seg = nib.load(self.pairs[0][1]).get_fdata().squeeze()
        maxseg = seg.max()
        self.max_label_index = int(maxseg)


    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, index):
        # moving, moving_seg, fixed, fixed_seg = load_pair(self.pairs[index])
        # print(self.pairs[index])
        moving, moving_seg = load_pair(self.pairs[index][0], self.pairs[index][1], self.isotropic, self.crop)
        fixed, fixed_seg = load_pair(self.pairs[index][2], self.pairs[index][3], self.isotropic, self.crop)
        # create torch tensor
        fixed, moving = torch.from_numpy(np.ascontiguousarray(fixed)).float(), torch.from_numpy(np.ascontiguousarray(moving)).float()
        fixed_seg, moving_seg = torch.from_numpy(np.ascontiguousarray(fixed_seg)).long(), torch.from_numpy(np.ascontiguousarray(moving_seg)).long()
        ret = {
            'source_img': fixed[None],
            'target_img': moving[None],
            'source_label': fixed_seg[None],
            'target_label': moving_seg[None],
            'source_img_id': self.pair_ids[index][1],
            'target_img_id': self.pair_ids[index][0],
        }
        return ret

if __name__ == '__main__':
    # print(len(dataset))
    # print(dataset[0])
    # for i in dataset[0]:
    #     print(i.shape)
    for dataset in ['MGH10', 'CUMC12', 'IBSR18', 'LPBA40']:
        dataset = KleinDatasets(dataset=dataset, dry_run=True)
        m, f, ms, fs = dataset[0]
        print(torch.unique(ms) == torch.unique(fs))
        print(f.min(), f.max(), m.min(), m.max())
        # for m, ms, f, fs in (dataset.pairs):
        #     # print(m, f)
        #     print(ms, fs)
        # print()