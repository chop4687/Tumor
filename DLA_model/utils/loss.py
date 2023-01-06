import torch
from torch import nn
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

def focal_loss(pred, x_cen,y_cen,sig=1):
    ''' Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
    '''
    batch = pred.shape[0]
    pred = pred.squeeze()
    gt = torch.zeros((batch,22,44)).to(0)
    all = torch.zeros((2,22,44)).to(0)
    x = torch.arange(1,23).to(0)
    y = torch.arange(1,45).to(0)
    for i in range(44):
      all[0,:,i] += x
    for i in range(22):
      all[1,i,:] += y
    for i in range(batch):
        gt[i] = torch.exp(-1 * ((all[0,...]-x_cen[i])**2 + (all[1,...]-y_cen[i])**2)/(2*sig**2))
    loss = (1-gt)**2 * pred**4 * torch.log(1-pred)
    for i in range(batch):
        loss[i,x_cen[i],y_cen[i]] = (1-pred[i,x_cen[i],y_cen[i]])**4 * torch.log(pred[i,x_cen[i],y_cen[i]])
    loss = -loss.sum() / batch
    return loss


class MultiTaskLoss(nn.Module):
    def __init__(self, task_num, model):
        super(MultiTaskLoss, self).__init__()
        self.model = model
        self.task_num = task_num
        self.log_vars = nn.Parameter(torch.zeros((task_num)))
        self.CE = nn.CrossEntropyLoss()
    def forward(self, input, densenet, targets, mask):
        predict_img, predict_patch = self.model(input, densenet)

        precision1 = torch.exp(-self.log_vars[0])
        loss = torch.sum(precision1 * self.CE(predict_img, targets.long()) + self.log_vars[0], -1)

        precision2 = torch.exp(-self.log_vars[1])
        loss += torch.sum(precision2 * Dice_loss(predict_patch,mask,targets) + self.log_vars[1], -1)

        loss = torch.mean(loss)

        return loss, predict_img, predict_patch, self.log_vars.data.tolist()
