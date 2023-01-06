import torch

def Dice_loss(input,target,class_y):
    top = torch.sum(input * target,dim=(1,2,3)) + 1e-10
    down = torch.sum(input,dim=(1,2,3)) + torch.sum(target, dim = (1,2,3)) + 1e-10
    loss = 2 * top / down
    loss = class_y * (1 - loss)
    loss = loss.sum() / class_y.sum()
    return loss

def IoU_loss(input,target):
    top = torch.sum(input * target,dim=(1,2,3))
    down = torch.sum(input,dim=(1,2,3)) + torch.sum(target, dim = (1,2,3)) - top
    loss = top / down
    loss = 1 - loss
    return loss.mean()

def L1_loss(input,target):
    loss = torch.abs(input - target)**2
    loss = torch.sum(loss, dim=(1,2,3))
    loss = loss.mean()
    return loss
