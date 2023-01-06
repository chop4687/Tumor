import random
from torch import nn,where
from utils.transform import *
import numpy as np

def to_one_hot_vector(num_class, label):
    vec = torch.zeros((label.shape[0], num_class)).to(0)
    # vec[torch.arange(label.shape[0]), label] = soft
    # vec[:,0] = 1 - soft
    return vec

def mixup(input, target, mask,batch_size,name):
    ### input은 48,3,750,1500
    if batch_size % 2 == 1 and batch_size != 1:
        batch_size -= 1
        input = input[:input.shape[0]-1,...]
        target = target[:target.shape[0]-1,...]
        mask = mask[:mask.shape[0]-1,...]
        name = name[:len(name)-1]
        #zero_sum = zero_sum[:zero_sum.shape[0]-1]
    ## target은 48
    mix_target = torch.zeros((batch_size)).to(0)
    normal_idx = torch.where(target == 0)[0]
    tumor_idx = torch.where(target != 0)[0]
    if len(normal_idx) == 0 or len(tumor_idx) == 0:
        return input, target, mask, mix_target, name
    ############### mix up func
    normal_idx = np.random.choice(normal_idx.cpu(),batch_size//2)
    tumor_idx = np.random.choice(tumor_idx.cpu(),batch_size//2)
    total_name = np.concatenate((np.array(name)[tumor_idx],np.array(name)[normal_idx]))

    temp = (input[tumor_idx,...])*(mask[tumor_idx,...] == 1).float()
    hole_input = (input[normal_idx,...])*(mask[tumor_idx,...] != 1).float()
    mix_input = hole_input + temp
    origin_input = input[:batch_size//2,...]
    total_input = torch.cat((origin_input, mix_input))
    ###############여기까지 input 관련
    ###################여기부터 soft target, hard target 만들기
    ori_hard_target = target[:batch_size//2]
    zero_mask = mask[:batch_size//2]
    # ori_soft_target = 1 - (mask[:batch_size//2].sum(dim=(1,2,3)) / (750*1500-zero_sum[:batch_size//2]))
    mix_hard_target = target[tumor_idx]
    # mix_soft_target = 1 - (mask[tumor_idx].sum(dim=(1,2,3)) / (750*1500-zero_sum[tumor_idx]))
    total_hard_target = torch.cat((ori_hard_target, mix_hard_target))
    # total_soft_target = torch.cat((ori_soft_target, mix_soft_target))
    ################ segmentation쪽도 손봐야됨
    ori_mask = mask[:batch_size//2]
    mix_mask = mask[tumor_idx]
    total_mask = torch.cat((ori_mask, mix_mask))

    mix_target[tumor_idx] += 1
    ##########################################
    rand_idx = torch.randperm(batch_size)
    total_input = total_input[rand_idx,...]
    total_hard_target = total_hard_target[rand_idx]

    total_mask = total_mask[rand_idx]
    mix_target = mix_target[rand_idx]
    name = total_name[rand_idx]
    return total_input, total_hard_target, total_mask, mix_target, name
