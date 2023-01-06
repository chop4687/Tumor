from torch import nn

def cls_loss(input, target, idx):
    result = 0.0
    batch = len(input)
    for bat in range(batch):
        result += nn.CrossEntropyLoss()(input[bat,idx[bat,0:40],:],target[bat,idx[bat,0:40]].long())
    return result / batch

def reg_loss(input, target, idx, cls):
    result = 0.0
    batch = len(input)
    for bat in range(batch):
        for i in range(40):
            temp = cls[bat,idx[bat,i]]
            result += (temp * nn.SmoothL1Loss()(input[bat,idx[bat,i],:],target[bat,:]))
    return result / 40
