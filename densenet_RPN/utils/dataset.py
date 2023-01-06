from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import random
from utils.etc import find_box, find_box_binary
import torch
import numpy as np

class eight_patch(Dataset):
    def __init__(self, root, img_size = 375, transform = None, mode = 'train'):
        self.root = root
        self.img_size = img_size
        self.transform = transform
        self.file_tumor = os.listdir(os.path.join(root,'dataset/tumor'))

        if mode == 'train':
            #self.file_name = self.file_tumor[:int(len(self.file_tumor)*0.75)]
            self.file_name = self.file_tumor[int(len(self.file_tumor)*0.25):]
        elif mode == 'val':
            self.file_name = self.file_tumor[int(len(self.file_tumor)*0.15):int(len(self.file_tumor)*0.25)]
        else:
            self.file_name = self.file_tumor[:int(len(self.file_tumor)*0.15)]

        random.shuffle(self.file_name)
        self.input, self.bbox, self.mask = [], [], []

        for i in self.file_name:
            mask_img = Image.open(root+'mask_img/'+i)
            black_img = Image.open(root+'black/'+i)
            x_min, y_min, x_max, y_max = find_box_binary(black_img)
            for y in range(2):
                for x in range(4):
                    temp = mask_img.crop((x * self.img_size, y * self.img_size,
                                        (x + 1) * self.img_size, (y + 1) * self.img_size))
                    temp_black = black_img.crop((x * self.img_size, y * self.img_size,
                                        (x + 1) * self.img_size, (y + 1) * self.img_size))

                    x1_bar, y1_bar, x2_bar, y2_bar = find_box_binary(temp_black)
                    self.input.append(temp)
                    self.bbox.append(torch.FloatTensor([x1_bar,y1_bar,x2_bar,y2_bar]))
                    self.mask.append(temp_black)

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        input, bbox, mask = self.input[idx], self.bbox[idx], self.mask[idx]
        if self.transform is not None:
            input = self.transform(input)
            mask = self.transform(mask)

        return input, bbox, mask


class seg_patch(Dataset):
    def __init__(self, root, img_size = 375, transform = None, mode = 'train'):
        self.root = root
        self.img_size = img_size
        self.transform = transform
        self.file_tumor = os.listdir(os.path.join(root,'dataset/tumor'))

        if mode == 'train':
            #self.file_name = self.file_tumor[:int(len(self.file_tumor)*0.75)]
            self.file_name = self.file_tumor[int(len(self.file_tumor)*0.25):]
        elif mode == 'val':
            self.file_name = self.file_tumor[int(len(self.file_tumor)*0.15):int(len(self.file_tumor)*0.25)]
        else:
            self.file_name = self.file_tumor[:int(len(self.file_tumor)*0.15)]

        random.shuffle(self.file_name)
        self.input, self.bbox, self.mask = [], [], []

        for i in self.file_name:
            mask_img = Image.open(root+'mask_img/' + i)
            black_img = Image.open(root+'tumor_segmentation/' + i)
            idx = np.load(root+'bbox_idx/' + i[:-4] + '.npy')
            rand_line = random.randint(1,len(idx))
            x_min, y_min, x_max, y_max = idx[rand_line-1,0],idx[rand_line-1,1],idx[rand_line-1,2],idx[rand_line-1,3]
            #x_min, y_min, x_max, y_max = find_box_binary(black_img)
            for l in range(3):
                x = random.randint(x_min, x_max)
                y = random.randint(y_min, y_max)
                x = max(0,min(x,750))
                y = max(0,min(y,1500))
                temp = mask_img.crop((y-187, x-187, y+188, x+188))
                temp_black = black_img.crop((y-187, x-187, y+188, x+188))

                x1_bar, y1_bar, x2_bar, y2_bar = find_box_binary(temp_black)
                black = torch.zeros((1,375,375))
                black[:,x1_bar:x2_bar,y1_bar:y2_bar] = 1
                self.input.append(temp)
                self.bbox.append(torch.FloatTensor([x1_bar,y1_bar,x2_bar,y2_bar]))
                self.mask.append(black)
            for l in range(1):
                x = random.randint(0, 750)
                y = random.randint(0, 1500)
                temp = mask_img.crop((y-187, x-187, y+188, x+188))
                temp_black = black_img.crop((y-187, x-187, y+188, x+188))

                x1_bar, y1_bar, x2_bar, y2_bar = find_box_binary(temp_black)
                black = torch.zeros((1,375,375))
                black[:,x1_bar:x2_bar,y1_bar:y2_bar] = 1
                self.input.append(temp)
                self.bbox.append(torch.FloatTensor([x1_bar,y1_bar,x2_bar,y2_bar]))
                self.mask.append(black)

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        input, bbox, mask = self.input[idx], self.bbox[idx], self.mask[idx]
        if self.transform is not None:
            input = self.transform(input)
            #mask = self.transform(mask)

        return input, bbox, mask
