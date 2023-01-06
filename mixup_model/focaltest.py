import torch
from torch import nn

from utils.loss import FocalLoss

def scale(ten):
    return (ten - ten.min()) / (ten.max() - ten.min())
if __name__ == '__main__':
    a = scale(torch.empty((48,4)))
    b = torch.ones((48,4))
    CE = nn.CrossEntropyLoss()
    BCE = nn.BCELoss()
    focal = FocalLoss(gamma=1)
    print(BCE(a,b))
    print(focal(a,b))
