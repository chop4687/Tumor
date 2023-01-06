import torch
import torch.nn as nn
import os
from torchvision import transforms as T
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from visdom import Visdom
import numpy as np
from matplotlib import pyplot as plt
import torch.nn.functional as F
import random

class Resize(T.Resize):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, x):
        img , target = x
        """
        Args:
            img (PIL Image): Image to be scaled.

        Returns:
            PIL Image: Rescaled image.
        """
        return T.functional.resize(img, self.size, self.interpolation), T.functional.resize(target, self.size, self.interpolation)

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)


class RandomSwap():
    def __init__(self, rep):
        self.rep = rep

    def __call__(self, x):
        img, target = x
        img = T.ToTensor()(img)
        target = T.ToTensor()(target)
        for i in range(self.rep):
            idx_x1 = random.randint(0,img.shape[1]-25)
            idx_y1 = random.randint(0,img.shape[2]-25)

            idx_x2 = random.randint(0,img.shape[1]-25)
            idx_y2 = random.randint(0,img.shape[2]-25)

            patch = img[:,idx_x1:idx_x1+25,idx_y1:idx_y1+25]
            temp = torch.empty_like(patch)
            temp.copy_(patch)

            other = img[:,idx_x2:idx_x2+25,idx_y2:idx_y2+25]
            patch.copy_(other)
            other.copy_(temp)
            # img[:,idx_x1:idx_x1+100,idx_y1:idx_y1+100] = img[:,idx_x2:idx_x2+100,idx_y2:idx_y2+100]
            # img[:,idx_x2:idx_x2+100,idx_y2:idx_y2+100] = temp

        return T.ToPILImage()(img),T.ToPILImage()(target)

class RandomHorizontalFlip(T.RandomHorizontalFlip):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self,x):
        img,target = x
        if random.random() < self.p:
            return T.functional.hflip(img), T.functional.hflip(target)
        return img, target

class RandomCrop(T.RandomCrop):
    def __init__(
        self,
        size=None,
        padding=None,
        pad_if_needed=False,
        fill=0,
        padding_mode='constant'
    ):
        super(RandomCrop, self).__init__(
            size=size,
            padding=padding,
            pad_if_needed=pad_if_needed,
            fill=fill,
            padding_mode=padding_mode
        )
    def __call__(self, x):
        img , target = x
        size = [int(img.size[1]*0.8),int(img.size[0]*0.8)]
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)
            target = F.pad(img, self.padding, self.fill, self.padding_mode)

        if self.pad_if_needed and img.size[0] < size[1]:
            img = F.pad(img, (size[1] - img.size[0], 0), self.fill, self.padding_mode)
            target = F.pad(target, (size[1] - target.size[0], 0), self.fill, self.padding_mode)
        if self.pad_if_needed and img.size[1] < size[0]:
            img = F.pad(img, (0, size[0] - img.size[1]), self.fill, self.padding_mode)
            target = F.pad(target, (0, size[0] - target.size[1]), self.fill, self.padding_mode)

        i, j, h, w = self.get_params(img, size)

        return T.functional.crop(img, i, j , h, w), T.functional.crop(target, i, j, h, w)


class RandomRotation(T.RandomRotation):
    def __init__(self,degrees,resample=False, expand=False, center=None):
        super(RandomRotation, self).__init__(
            degrees,
            resample=resample,
            expand=expand,
            center=center
        )

    def __call__(self,x):
        img,target = x
        angle = self.get_params(self.degrees)

        return T.functional.rotate(img, angle, self.resample, self.expand, self.center), T.functional.rotate(target, angle, self.resample, self.expand, self.center)
