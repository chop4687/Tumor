import os

import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.utils import save_image
import random
from utils.dataset import tumor
from utils.transform import *
from visdom import Visdom
import torch.nn.functional as F
from torch import nn
import numpy as np
def scale(ten):
    return (ten - ten.min()) / (ten.max() - ten.min())

def to_one_hot_vector(num_class, label, soft):
    vec = torch.zeros((label.shape[0], num_class))
    vec[torch.arange(label.shape[0]), label] = soft
    vec[:,0] = 1 - soft
    return vec

if __name__ == '__main__':
    vis = Visdom(port = 13246, env = 'test')
    valset = tumor(root = '/home/junkyu/project/tumor_detection/mixup_model/data',transform = T.Compose([RandomRotation(5),
                                RandomVerticalFlip(),
                                RandomHorizontalFlip(),
                                ColorJitter(brightness=random.random(),
                                                       contrast=random.random(),
                                                       saturation=random.random(),
                                                       hue=random.random() / 2),
                                ToTensor(),
                                Normalize(mean = [0.2602], std = [0.2729])
                                ]),mode='val',cv=1)
    val_loader = DataLoader(valset, batch_size=48,shuffle=True,num_workers=2)
    for i,(input,target,mask,name) in enumerate(val_loader):
        normal_idx = torch.where(target == 0)[0]
        tumor_idx = torch.where(target != 0)[0]
        ############### mix up func
        normal_idx = np.random.choice(normal_idx,24)
        tumor_idx = np.random.choice(tumor_idx,24)

        vis.image(input[tumor_idx[0],...]*0.2729+0.2602)
        temp = (input[tumor_idx,...]*0.2729+0.2602)*(mask[tumor_idx,...] == 1).float()
        vis.image(temp[0,...])
        vis.image(input[normal_idx[0],...]*0.2729+0.2602)
        hole_input = (input[normal_idx,...]*0.2729+0.2602)*(mask[tumor_idx,...] != 1).float()
        mix_input = hole_input + temp
        vis.image(mix_input[0,...])
        # ori_input = input[:24,...]
        # ori_hard_target = target[:24]
        # ori_soft_target = mask[:24].sum(dim=(1,2,3)) / (750*1500)
        # ori_soft_target = to_one_hot_vector(4,ori_hard_target,ori_soft_target)
        # hard_target = target[tumor_idx]
        # mix_soft_target = mask[tumor_idx].sum(dim=(1,2,3)) / (750*1500)
        # mix_soft_target = to_one_hot_vector(4,ori_hard_target,mix_soft_target)
        # total_input = torch.cat((ori_input,mix_input))
        # total_hard_target = torch.cat((ori_hard_target,hard_target))
        # total_soft_target = torch.cat((ori_soft_target,soft_target))
        # rand_idx = torch.randperm(48)
        # total_input = total_input[rand_idx,...]
        # total_hard_target = total_hard_target[rand_idx,...]
        # total_soft_target = total_soft_target[rand_idx,...]
        # soft_target = mask[]
        exit()
