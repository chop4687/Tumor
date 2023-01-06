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
        if mode == 'train':
            self.file_name = np.concatenate((self.file_normal[normal_train_idx], self.file_tumor[tumor_train_idx]),axis=None)
        else:
            self.file_name = np.concatenate((self.file_normal[normal_test_idx], self.file_tumor[tumor_test_idx]),axis=None)
        ################################################################
        # if mode == 'train':
        #     self.file_name = self.file_normal[:int(len(self.file_normal)*0.75)] + self.file_tumor[:int(len(self.file_tumor)*0.75)]
        # elif mode == 'val':
        #     self.file_name = self.file_normal[int(len(self.file_normal)*0.75+1):int(len(self.file_normal)*0.85)] + self.file_tumor[int(len(self.file_tumor)*0.75):int(len(self.file_tumor)*0.85)]
        # else:
        #     self.file_name = self.file_normal[int(len(self.file_normal)*0.85):] + self.file_tumor[int(len(self.file_tumor)*0.85):]
        #
        # random.shuffle(self.file_name)

    def _load(self, file):
        file = [file]
        input,  class_y, masks, name, big_img = [], [], [], [], []
        for i in file:
            if i in self.file_normal:
                img = Image.open(os.path.join(self.root, 'dataset/normal', i)).convert('RGB').resize((1500,750))
                mask = Image.open(os.path.join(self.root, 'mask/normal', i[:-4]+'.png')).convert('RGB').resize((1500,750))
                mask_img = ImageChops.multiply(img, mask)
                black_img = T.ToPILImage()(torch.zeros((750,1500)))
                for y in range(2):
                    for x in range(4):
                        temp = mask_img.crop((x * self.img_size, y * self.img_size,
                                            (x + 1) * self.img_size, (y + 1) * self.img_size))
                        big_img.append(img)
                        input.append(temp)
                        class_y.append(0)
                        masks.append(black_img)
                        name.append(i)

            else:
                black_img = Image.open('/home/junkyu/project/tumor_detection/densenet_RPN/data/bbox_img_375/'+i[:-4]+'.png').convert('L')
                #x_min, y_min, x_max, y_max = find_box_binary(black_img)
                #black_img[x_min:x_max, y_min:y_max] = 1

                img = Image.open(os.path.join(self.root, 'dataset/tumor', i)).convert('RGB').resize((1500,750))
                mask = Image.open(os.path.join(self.root, 'mask/tumor', i[:-4]+'.png')).convert('RGB').resize((1500,750))
                mask_img = ImageChops.multiply(img, mask)
                #black_img = T.ToPILImage()(black_img)
                for y in range(2):
                    for x in range(4):
                        temp = mask_img.crop((x * self.img_size, y * self.img_size,
                                            (x + 1) * self.img_size, (y + 1) * self.img_size))

                        temp_black = black_img.crop((x * self.img_size, y * self.img_size,
                                            (x + 1) * self.img_size, (y + 1) * self.img_size))

                        big_img.append(img)
                        input.append(temp)
                        ######################################################
                        for t in ['Ameloblastoma','dentigerous_cyst','OKC']:
                            tumor_split = os.listdir('/home/junkyu/data/all_tumor/' + t)
                            if i in tumor_split:
                                if t == 'Ameloblastoma':
                                    class_y.append(1)
                                elif t == 'dentigerous_cyst':
                                    class_y.append(2)
                                else:
                                    class_y.append(3)
                        ######################################################
                        masks.append(black_img)
                        name.append(i)

        return big_img, input, class_y, masks, name

    def __len__(self):
        return len(self.file_name)

    def __getitem__(self, idx):
        file = self.file_name[idx]
        big_img, input, class_y, masks, name = self._load(file)
        if self.transform is not None:
            for i in range(8):
                big_img[i], input[i], masks[i] = self.transform((big_img[i],input[i],masks[i]))
        # input 8개씩
        return big_img, input, class_y, masks, name


# class patch(Dataset):
#     def __init__(self, root, img_size = 375, transform = None, mode = 'train'):
#         self.root = root
#         self.img_size = img_size
#         self.transform = transform
#         self.file_tumor = os.listdir(os.path.join(root,'dataset/tumor'))
#
#         if mode == 'train':
#             self.file_name = self.file_tumor[:int(len(self.file_tumor)*0.75)]
#         elif mode == 'val':
#             self.file_name = self.file_tumor[int(len(self.file_tumor)*0.75):int(len(self.file_tumor)*0.85)]
#         else:
#             self.file_name = self.file_tumor[int(len(self.file_tumor)*0.85):]
#
#         random.shuffle(self.file_name)
#         self.input, self.target, self.masks, self.name, self.position = [], [], [], [], []
#         for i in self.file_name:
#             anno = Image.open(os.path.join(self.root, 'tumor_bbox', i)).convert('RGB')
#             x_min, y_min, x_max, y_max = find_box(anno)
#             black_img = torch.zeros((1500,3000))
#             black_img[x_min:x_max, y_min:y_max] = 1
#             ran = (((x_max - x_min) * (y_max - y_min)) / 4) * 0.15
#
#             img = Image.open(os.path.join(self.root, 'dataset/tumor', i)).convert('RGB').resize((1500,750))
#             mask = Image.open(os.path.join(self.root, 'mask/tumor', i[:-4]+'.png')).convert('RGB').resize((1500,750))
#             mask_img = ImageChops.multiply(img, mask)
#             black_img = T.ToPILImage()(black_img).resize((1500,750))
#             for y in range(2):
#                 for x in range(4):
#                     temp = mask_img.crop((x * self.img_size, y * self.img_size,
#                                         (x + 1) * self.img_size, (y + 1) * self.img_size))
#
#                     temp_black = black_img.crop((x * self.img_size, y * self.img_size,
#                                         (x + 1) * self.img_size, (y + 1) * self.img_size))
#
#                     if T.ToTensor()(temp_black).sum() < ran:
#                         self.input.append(temp)
#                         self.target.append(0)
#                         self.masks.append(temp_black)
#                         self.name.append(i)
#                         self.position.append(x+(y*4))
#                     else:
#                         self.input.append(temp)
#                         self.target.append(1)
#                         self.masks.append(temp_black)
#                         self.name.append(i)
#                         self.position.append(x+(y*4))
#
#     def __len__(self):
#         return len(self.input)
#
#     def __getitem__(self, idx):
#         input, target, masks, name, position = self.input[idx], self.target[idx], self.masks[idx], self.name[idx], self.position[idx]
#         if self.transform is not None:
#             input = self.transform(input)
#             masks = self.transform(masks)
#
#         return input, target, masks, name, position
#
# class random_patch(Dataset):
#     def __init__(self, root, img_size = 375, transform = None, mode = 'train'):
#         self.root = root
#         self.img_size = img_size
#         self.transform = transform
#         self.file_tumor = os.listdir(os.path.join(root,'dataset/tumor'))
#
#         if mode == 'train':
#             self.file_name = self.file_tumor[:int(len(self.file_tumor)*0.75)]
#         elif mode == 'val':
#             self.file_name = self.file_tumor[int(len(self.file_tumor)*0.75):int(len(self.file_tumor)*0.85)]
#         else:
#             self.file_name = self.file_tumor[int(len(self.file_tumor)*0.85):]
#
#         random.shuffle(self.file_name)
#         self.input, self.target, self.masks, self.name = [], [], [], []
#
#         for i in self.file_name:
#             anno = Image.open(os.path.join(self.root, 'tumor_bbox', i)).convert('RGB')
#             x_min, y_min, x_max, y_max = find_box(anno)
#             black_img = torch.zeros((1500,3000))
#             black_img[x_min:x_max, y_min:y_max] = 1
#             ran = (((x_max - x_min) * (y_max - y_min)) / 4) * 0.15
#
#             img = Image.open(os.path.join(self.root, 'dataset/tumor', i)).convert('RGB').resize((1500,750))
#             mask = Image.open(os.path.join(self.root, 'mask/tumor', i[:-4]+'.png')).convert('RGB').resize((1500,750))
#             mask_img = ImageChops.multiply(img, mask)
#             black_img = T.ToPILImage()(black_img).resize((1500,750))
#             for l in range(3):
#                 x = random.randint(0,1500 - img_size)
#                 y = random.randint(0,750 - img_size)
#                 temp = mask_img.crop((x, y, x+img_size, y+img_size))
#                 temp_black = black_img.crop((x, y, x+img_size, y+img_size))
#                 block_size = T.ToTensor()(temp_black).sum()
#                 self.input.append(temp)
#                 self.target.append(block_size / img_size**2)
#                 self.masks.append(temp_black)
#                 self.name.append(i)
#
#
#     def __len__(self):
#         return len(self.input)
#
#     def __getitem__(self, idx):
#         input, target, masks, name = self.input[idx], self.target[idx], self.masks[idx], self.name[idx]
#         if self.transform is not None:
#             input, masks = self.transform((input,masks))
#
#         return input, target, masks, name
