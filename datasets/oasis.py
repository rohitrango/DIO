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

class OASISImageOnly(Dataset):
    ''' dataset that only returns images (consider transforms here) '''
    def __init__(self, data_root='data/oasis', split='train'):
        super().__init__()
        self.images = sorted(glob(osp.join(data_root, 'imagesTr' if split == 'train' else 'imagesTs', '*.nii.gz')))
        self.split = split
        self.N = len(self.images)

    def __len__(self):
        return self.N
    
    def __getitem__(self, index):
        output = {}
        source_img = self.images[index]
        # add metadata
        output['source_img_path'] = source_img
        # load source and optionally edit it (if its from the training split)
        src_img = nib.load(source_img).get_fdata()
        if self.split == 'train':
            if self.rng.rand() < 0.5:
                src_img = np.flip(src_img, axis=0) + 0
            if self.rng.rand() < 0.5:
                src_img = np.flip(src_img, axis=1) + 0
            if self.rng.rand() < 0.5:
                src_img = np.flip(src_img, axis=2) + 0
            # color jitter (add noise)
            transformed_img = src_img + self.rng.normal(0, 0.025, src_img.shape)
            transformed_img = np.clip(transformed_img, 0, 1) + 0
        else:
            transformed_img = src_img.copy()
        # put them in output
        output['source_img'] = torch.from_numpy(src_img).float()[None]*2 - 1
        output['transformed_img'] = torch.from_numpy(transformed_img).float()[None]*2 - 1
        return output

class OASIS(Dataset):
    ''' 
    This dataset is for the L2R submission.  
    It has 3 splits: train and val are the same split but val has a different fixed to moving mapping (i -> i+1 only) but train split has mapping (i -> j) for i~=j
    val and test have the mapping specified in L2R
    '''
    def __init__(self, data_root='data/oasis', split='train', seed=1685):
        super().__init__()
        self.data_root = data_root
        self.l2r_data = json.load(open(osp.join(data_root, 'OASIS_dataset.json'), 'r'))
        self.images = sorted(glob(osp.join(data_root, 'imagesTr' if split != 'test' else 'imagesTs', '*.nii.gz')))
        self.labels = sorted(glob(osp.join(data_root, 'labelsTr', '*.nii.gz'))) if split != 'test' else None
        if split == 'train':
            # delete the last 20 val images
            self.images = self.images[:-20]
            self.labels = self.labels[:-20]
        self.split = split
        self.N = len(self.images)
        self.max_label_index = 35   # the dataset is labeled from 0 to 35
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        if self.split == 'train':
            return self.N
        elif self.split == 'val':
            return len(self.l2r_data['registration_val'])
        else:
            return len(self.l2r_data['registration_test'])
    
    def _modulo(self, index):
        return index % self.N
    
    def __getitem__(self, index):
        '''
        WARNING: In my notation, I choose fixed = source and moving = target
        '''
        output = {}
        if self.split == 'train':
            targ_index = self._modulo(1 + self.rng.randint(self.N-1)) 
            source_img = self.images[index]
            source_label = self.labels[index] if self.split != 'test' else None
            target_img = self.images[targ_index] 
            target_label = self.labels[targ_index] if self.split != 'test' else None
        else:
            # val or test
            pair = self.l2r_data[f'registration_{self.split}'][index]
            source_img = osp.join(self.data_root, pair['fixed'])
            target_img = osp.join(self.data_root, pair['moving'])
            if self.split == 'val':
                source_label = source_img.replace("images", "labels")
                target_label = target_img.replace("images", "labels")
            else:
                source_label, target_label = None, None
            
        # add metadata
        output['source_img_path'] = source_img
        output['target_img_path'] = target_img
        output['source_label_path'] = source_label if source_label is not None else 0
        output['target_label_path'] = target_label if target_label is not None else 0
        # load source and target
        src_img = nib.load(source_img).get_fdata()
        tgt_img = nib.load(target_img).get_fdata()
        if self.split != 'test':
            src_label = nib.load(source_label).get_fdata()
            tgt_label = nib.load(target_label).get_fdata()
        else:
            src_label, tgt_label = None, None
        # if train, do some random flipping
        # if self.split == 'train':
        #     for ax in range(3):
        #         if self.rng.rand() < 0.5:
        #             src_img = np.flip(src_img, axis=ax) + 0
        #             tgt_img = np.flip(tgt_img, axis=ax) + 0
        #             if src_label is not None:
        #                 src_label = np.flip(src_label, axis=ax) + 0
        #                 tgt_label = np.flip(tgt_label, axis=ax) + 0
        # put them in output
        output['source_img'] = torch.from_numpy(src_img).float()[None]# *2 - 1
        output['target_img'] = torch.from_numpy(tgt_img).float()[None]# *2 - 1
        output['source_label'] = torch.from_numpy(src_label).long()[None] if src_label is not None else 0
        output['target_label'] = torch.from_numpy(tgt_label).long()[None] if tgt_label is not None else 0
        return output

class OASISNeurite(Dataset):
    '''
    neurite version of the dataset
    '''
    def __init__(self, data_root='data/neurite-oasis', split='train', seed=1685, is3d=True, seg4=False):
        super().__init__()
        self.data_root = data_root
        if split != 'test':
            seg = "24" if not seg4 else "4"
            self.images = sorted(glob(osp.join(data_root, 'OASIS*MR1', 'aligned_norm.nii.gz' if is3d else 'slice_norm.nii.gz')))
            self.labels = sorted(glob(osp.join(data_root, 'OASIS*MR1', 'aligned_seg35.nii.gz' if is3d else f'slice_seg{seg}.nii.gz')))
            if split == 'train':
                self.images = self.images[:-20]
                self.labels = self.labels[:-20]
            else:
                self.images = self.images[-20:]
                self.labels = self.labels[-20:]
        else:
            if not is3d:
                raise NotImplementedError("no test dataset for 2d")
            self.images = sorted(glob(osp.join(data_root, 'test', 'img*norm.nii.gz')))
            self.labels = None
        # save extra args 
        self.split = split
        self.N = len(self.images) - (0 if self.split == 'train' else 1)
        self.max_label_index = 35 if is3d else 24   # the dataset is labeled from 0 to 35
        self.is3d = is3d
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return self.N

    def _modulo(self, index):
        return index % self.N
    
    def __getitem__(self, index):
        '''
        WARNING: In my notation, I choose fixed = source and moving = target
        '''
        output = {}
        if self.split == 'train':
            targ_index = self._modulo(1 + self.rng.randint(self.N-1)) 
        else:
            targ_index = index + 1

        source_img = self.images[index]
        source_label = self.labels[index] if self.split != 'test' else None
        target_img = self.images[targ_index] 
        target_label = self.labels[targ_index] if self.split != 'test' else None
            
        # add metadata
        output['source_img_path'] = source_img
        output['target_img_path'] = target_img
        output['source_label_path'] = source_label if source_label is not None else 0
        output['target_label_path'] = target_label if target_label is not None else 0
        # load source and target
        src_img = nib.load(source_img).get_fdata().squeeze()
        tgt_img = nib.load(target_img).get_fdata().squeeze()
        if self.split != 'test':
            src_label = nib.load(source_label).get_fdata().squeeze()
            tgt_label = nib.load(target_label).get_fdata().squeeze()
        else:
            src_label, tgt_label = None, None
        # if train, do some random flipping
        # if self.split == 'train':
        #     for ax in range(3):
        #         if self.rng.rand() < 0.5:
        #             src_img = np.flip(src_img, axis=ax) + 0
        #             tgt_img = np.flip(tgt_img, axis=ax) + 0
        #             if src_label is not None:
        #                 src_label = np.flip(src_label, axis=ax) + 0
        #                 tgt_label = np.flip(tgt_label, axis=ax) + 0
        # put them in output
        output['source_img'] = torch.from_numpy(src_img).float()[None]# *2 - 1
        output['target_img'] = torch.from_numpy(tgt_img).float()[None]# *2 - 1
        output['source_label'] = torch.from_numpy(src_label).long()[None] if src_label is not None else 0
        output['target_label'] = torch.from_numpy(tgt_label).long()[None] if tgt_label is not None else 0
        return output

class OASISNeurite3D(OASISNeurite):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, is3d=True)

class OASISNeurite2D(OASISNeurite):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, is3d=False)

if __name__ == '__main__':
    # dataset = OASIS('../data/oasis', split='val')
    # # dataset = OASISImageOnly('../data/oasis', split='train')
    # print(len(dataset))
    # for i in range(len(dataset)):
    #     if i == 4:
    #         break
    #     datum = dataset[i]
    #     for k, v in datum.items():
    #         if isinstance(v, torch.Tensor):
    #             print(k, v.shape, v.min(), v.max())
    #         else:
    #             print(k, v)
    #     # pprint.pprint(dataset[i])
    #     print(datum['source_img_path'].split('/')[-1].split(".")[0].split("_")[-2])
    #     print(datum['target_img_path'].split('/')[-1].split(".")[0].split("_")[-2])
    #     print()
    #     # Image.fromarray((datum['source_img'].numpy().squeeze()[80]*0.5 + 0.5)*255).convert('RGB').save(f'./test_{i}.png')
    #     # Image.fromarray((datum['transformed_img'].numpy().squeeze()[80]*0.5 + 0.5)*255).convert('RGB').save(f'./test_{i}_transformed.png')

    dataset = OASISNeurite3D()
    datum = dataset[0]
    for k, v in datum.items():
        print(k, v.shape if isinstance(v, torch.Tensor) else v)

    dataset = OASISNeurite2D()
    datum = dataset[0]
    for k, v in datum.items():
        print(k, v.shape if isinstance(v, torch.Tensor) else v)

    # for split, is3d in [['train', True], ['val', True], ['test',  True], ['train', False]]:
    #     dataset = OASISNeurite(split=split, is3d=is3d)
    #     print(split, is3d, len(dataset))
    #     for i in range(len(dataset)):
    #         datum = dataset[i]
    #         for k, v in datum.items():
    #             if isinstance(v, torch.Tensor):
    #                 print(k, v.shape)
    #             else:
    #                 print(k, v)
    #         print()
    #         if i >= 2:
    #             break
    #     print()
