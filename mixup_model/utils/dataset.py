import torch
import torch.nn as nn
import os
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageChops
import random
from .etc import find_box_binary
from sklearn.model_selection import KFold
import numpy as np

class tumor(Dataset):
    def __init__(self, root, img_size = 375, transform = None, mode = 'train',cv = 0):
        self.root = root
        self.img_size = img_size
        self.transform = transform
        self.file_normal = np.array(os.listdir(os.path.join(root,'dataset/normal')))
        self.file_tumor = np.array(os.listdir(os.path.join(root,'dataset/tumor')))

        ################################################################
        # cross validation##############################################
        kf = KFold(n_splits=5)
        for i, (normal_train_idx, normal_test_idx) in enumerate(kf.split(self.file_normal)):
            if i == cv:
                break

        for i, (tumor_train_idx, tumor_test_idx) in enumerate(kf.split(self.file_tumor)):
            if i == cv:
                break
        # random.shuffle(normal_train_idx)
        # random.shuffle(tumor_train_idx)
        if mode == 'train':
            normal_train_idx = normal_train_idx[:int(len(normal_train_idx)*0.9)]
            tumor_train_idx = tumor_train_idx[:int(len(tumor_train_idx)*0.9)]
            ################################ boost
            self.file_tumor_boost = self.file_tumor[tumor_train_idx]
            for i in self.file_tumor_boost:
                for t in ['Ameloblastoma','dentigerous_cyst','OKC']:
                    tumor_split = os.listdir('/home/junkyu/data/all_tumor/' + t)
                    if i in tumor_split:
                        if t == 'Ameloblastoma':
                            self.file_tumor_boost = np.append(self.file_tumor_boost, i)
                            self.file_tumor_boost = np.append(self.file_tumor_boost, i)
                        elif t == 'OKC':
                            self.file_tumor_boost = np.append(self.file_tumor_boost, i)
            ################################
            self.file_name = np.concatenate((self.file_normal[normal_train_idx], self.file_tumor_boost),axis=None)
        elif mode == 'val':
            normal_train_idx = normal_train_idx[int(len(normal_train_idx)*0.9):]
            tumor_train_idx = tumor_train_idx[int(len(tumor_train_idx)*0.9):]
            ################################ boost
            self.file_tumor_boost = self.file_tumor[tumor_train_idx]
            for i in self.file_tumor_boost:
                for t in ['Ameloblastoma','dentigerous_cyst','OKC']:
                    tumor_split = os.listdir('/home/junkyu/data/all_tumor/' + t)
                    if i in tumor_split:
                        if t == 'Ameloblastoma':
                            self.file_tumor_boost = np.append(self.file_tumor_boost, i)
                            self.file_tumor_boost = np.append(self.file_tumor_boost, i)
                        elif t == 'OKC':
                            self.file_tumor_boost = np.append(self.file_tumor_boost, i)
            ################################
            self.file_name = np.concatenate((self.file_normal[normal_train_idx], self.file_tumor_boost),axis=None)
        else:
            self.file_name = np.concatenate((self.file_normal[normal_test_idx], self.file_tumor[tumor_test_idx]),axis=None)
    def _load(self, file):
        file = [file]
        for i in file:
            if i in self.file_normal:
                img = Image.open(os.path.join(self.root, 'dataset/normal', i)).convert('RGB').resize((1500,750))
                mask = Image.open(os.path.join(self.root, 'mask/normal', i[:-4]+'.png')).convert('RGB').resize((1500,750))
                mask_img = ImageChops.multiply(img, mask)
                black_img = T.ToPILImage()(torch.zeros((750,1500)))
                input = mask_img
                target = 0
                masks = black_img
                name = self.root+'/dataset/normal/'+i
                # for y in range(2):
                #     for x in range(4):
                #         temp = mask_img.crop((x * self.img_size, y * self.img_size,
                #                             (x + 1) * self.img_size, (y + 1) * self.img_size))
                #         input = temp)

            else:
                black_img = Image.open('/home/junkyu/project/tumor_detection/two_model/data/tumor_segmentation/'+i[:-4]+'.png').convert('L')
                img = Image.open(os.path.join(self.root, 'dataset/tumor', i)).convert('RGB').resize((1500,750))
                mask = Image.open(os.path.join(self.root, 'mask/tumor', i[:-4]+'.png')).convert('RGB').resize((1500,750))
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
                ######################################################
                masks = black_img
                name = self.root+'/dataset/tumor/'+i
                input = mask_img

        return input, target, masks, name

    def __len__(self):
        return len(self.file_name)

    def __getitem__(self, idx):
        file = self.file_name[idx]
        input, target, masks, name = self._load(file)
        if self.transform is not None:
            input, masks = self.transform((input,masks))
        return input, target, masks, name

class test_tumor(Dataset):
    def __init__(self, root, img_size = 375, transform = None, mode = 'train',cv = 0):
        self.root = root
        self.img_size = img_size
        self.transform = transform
        self.file_normal = np.array(os.listdir(os.path.join(root,'dataset/normal')))
        self.file_tumor = np.array(os.listdir(os.path.join(root,'dataset/tumor')))

        ################################################################
        # cross validation##############################################
        kf = KFold(n_splits=5)
        for i, (normal_train_idx, normal_test_idx) in enumerate(kf.split(self.file_normal)):
            if i == cv:
                break

        for i, (tumor_train_idx, tumor_test_idx) in enumerate(kf.split(self.file_tumor)):
            if i == cv:
                break
        # random.shuffle(normal_train_idx)
        # random.shuffle(tumor_train_idx)
        if mode == 'train':
            normal_train_idx = normal_train_idx[:int(len(normal_train_idx)*0.9)]
            tumor_train_idx = tumor_train_idx[:int(len(tumor_train_idx)*0.9)]
            ################################ boost
            self.file_tumor_boost = self.file_tumor[tumor_train_idx]
            for i in self.file_tumor_boost:
                for t in ['Ameloblastoma','dentigerous_cyst','OKC']:
                    tumor_split = os.listdir('/home/junkyu/data/all_tumor/' + t)
                    if i in tumor_split:
                        if t == 'Ameloblastoma':
                            self.file_tumor_boost = np.append(self.file_tumor_boost, i)
                            self.file_tumor_boost = np.append(self.file_tumor_boost, i)
                        elif t == 'OKC':
                            self.file_tumor_boost = np.append(self.file_tumor_boost, i)
            ################################
            self.file_name = np.concatenate((self.file_normal[normal_train_idx], self.file_tumor_boost),axis=None)
        elif mode == 'val':
            normal_train_idx = normal_train_idx[int(len(normal_train_idx)*0.9):]
            tumor_train_idx = tumor_train_idx[int(len(tumor_train_idx)*0.9):]
            ################################ boost
            self.file_tumor_boost = self.file_tumor[tumor_train_idx]
            for i in self.file_tumor_boost:
                for t in ['Ameloblastoma','dentigerous_cyst','OKC']:
                    tumor_split = os.listdir('/home/junkyu/data/all_tumor/' + t)
                    if i in tumor_split:
                        if t == 'Ameloblastoma':
                            self.file_tumor_boost = np.append(self.file_tumor_boost, i)
                            self.file_tumor_boost = np.append(self.file_tumor_boost, i)
                        elif t == 'OKC':
                            self.file_tumor_boost = np.append(self.file_tumor_boost, i)
            ################################
            self.file_name = np.concatenate((self.file_normal[normal_train_idx], self.file_tumor_boost),axis=None)
        else:
            self.file_name = np.concatenate((self.file_normal[normal_test_idx], self.file_tumor[tumor_test_idx]),axis=None)
    def _load(self, file):
        file = [file]
        for i in file:
            if i in self.file_normal:
                img = Image.open(os.path.join(self.root, 'dataset/normal', i)).convert('RGB').resize((1500,750))
                mask = Image.open(os.path.join(self.root, 'mask/normal', i[:-4]+'.png')).convert('RGB').resize((1500,750))
                mask_img = ImageChops.multiply(img, mask)
                black_img = T.ToPILImage()(torch.zeros((750,1500)))
                input = mask_img
                target = 0
                masks = black_img
                name = i
                # for y in range(2):
                #     for x in range(4):
                #         temp = mask_img.crop((x * self.img_size, y * self.img_size,
                #                             (x + 1) * self.img_size, (y + 1) * self.img_size))
                #         input = temp)

            else:
                black_img = Image.open('/home/junkyu/project/tumor_detection/two_model/data/tumor_segmentation/'+i[:-4]+'.png').convert('L')
                img = Image.open(os.path.join(self.root, 'dataset/tumor', i)).convert('RGB').resize((1500,750))
                mask = Image.open(os.path.join(self.root, 'mask/tumor', i[:-4]+'.png')).convert('RGB').resize((1500,750))
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
                ######################################################
                masks = black_img
                name = i
                input = mask_img

        return input, target, masks, name

    def __len__(self):
        return len(self.file_name)

    def __getitem__(self, idx):
        file = self.file_name[idx]
        input, target, masks, name = self._load(file)
        if self.transform is not None:
            input, masks = self.transform((input,masks))
        return input, target, masks, name
