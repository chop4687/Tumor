import os
from torchvision import transforms as T
from torch.utils.data import Dataset
from PIL import Image, ImageChops

class binary_classification(Dataset):
    def __init__(self,root,transform = None, mode = False):
        self.root = root
        self.transform = transform
        self.AB_name = os.listdir(root+'dataset/AB')
        self.DC_name = os.listdir(root+'dataset/DC')
        self.OKC_name = os.listdir(root+'dataset/OKC')
        self.NORMAL_name = os.listdir(root+'dataset/NORMAL')
        if mode:
            self.AB_name = self.AB_name[int(len(self.AB_name)*0.8):]
            self.DC_name = self.DC_name[int(len(self.DC_name)*0.8):]
            self.OKC_name = self.OKC_name[int(len(self.OKC_name)*0.8):]
            self.NORMAL_name = self.NORMAL_name[int(len(self.NORMAL_name)*0.8):]
        else:
            self.AB_name = self.AB_name[:int(len(self.AB_name)*0.8)]
            self.DC_name = self.DC_name[:int(len(self.DC_name)*0.8)]
            self.OKC_name = self.OKC_name[:int(len(self.OKC_name)*0.8)]
            self.NORMAL_name = self.NORMAL_name[:int(len(self.NORMAL_name)*0.8)]


        self.file_name = self.AB_name + self.DC_name + self.OKC_name + self.NORMAL_name
    def __len__(self):
        return len(self.file_name)

    def _load(self,file):
        if file in self.AB_name:
            img = Image.open(self.root+'dataset/AB/'+file).convert('RGB').resize((1500,750))
            mask = Image.open(self.root+'mask/AB/'+file[:-4]+'.png').convert('RGB').resize((1500,750))
            mask_img = ImageChops.multiply(img, mask)
            target = 1
        if file in self.DC_name:
            img = Image.open(self.root+'dataset/DC/'+file).convert('RGB').resize((1500,750))
            mask = Image.open(self.root+'mask/DC/'+file[:-4]+'.png').convert('RGB').resize((1500,750))
            mask_img = ImageChops.multiply(img, mask)
            target = 2
        if file in self.OKC_name:
            img = Image.open(self.root+'dataset/OKC/'+file).convert('RGB').resize((1500,750))
            mask = Image.open(self.root+'mask/OKC/'+file[:-4]+'.png').convert('RGB').resize((1500,750))
            mask_img = ImageChops.multiply(img, mask)
            target = 3
        if file in self.NORMAL_name:
            img = Image.open(self.root+'dataset/NORMAL/'+file).convert('RGB').resize((1500,750))
            mask = Image.open(self.root+'mask/NORMAL/'+file[:-4]+'.png').convert('RGB').resize((1500,750))
            mask_img = ImageChops.multiply(img, mask)
            target = 0

        return mask_img, target

    def __getitem__(self, idx):
        file = self.file_name[idx]
        input, target = self._load(file)
        if self.transform is not None:
            input = self.transform(input)
        return input, target
