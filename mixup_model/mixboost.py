import torch
import torch.nn as nn
import os
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageChops
import random
from sklearn.model_selection import KFold
import numpy as np

class boost(Dataset):
    def __init__(self, root, img_size = 375, transform = None):
        self.img = root
        self.img_size = img_size
        self.transform = transform

    def _load(self, file):
        mask_idx = file[:54] + 'mask' + file[61:]
        if file[62:68] == 'normal':
            black_img = T.ToPILImage()(torch.zeros((750,1500)))
            i = file[69:]
        else:
            i = file[68:]
            black_img = Image.open(file[:54] + 'tumor_segmentation/'+i[:-4]+'.png').convert('L')
        img = Image.open(file).convert('RGB').resize((1500,750))
        mask = Image.open(mask_idx[:-4]+'.png').convert('RGB').resize((1500,750))
        mask_img = ImageChops.multiply(img, mask)
        ######################################################
        for t in ['Ameloblastoma','dentigerous_cyst','OKC']:
            tumor_split = os.listdir('/home/junkyu/data/all_tumor/' + t)
            if i in tumor_split:
                if t == 'Ameloblastoma':
                    target = 1
                elif t == 'dentigerous_cyst':
                    target = 2
                else:
                    target = 3
            else:
                target = 0
        ######################################################
            masks = black_img
            input = mask_img

        return input, target, masks

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        file = self.img[idx]
        input, target, masks = self._load(file)
        if self.transform is not None:
            input, masks = self.transform((input,masks))
        return input, target, masks
