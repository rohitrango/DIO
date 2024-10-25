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

class AbdomenCTMR(Dataset):
    def __init__(self) -> None:
        super().__init__()

if __name__ == '__main__':
    pass