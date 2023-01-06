from PIL import Image
import torch
import os
from torchvision import transforms as T
from visdom import Visdom
import torch.distributions as tdist
def nor(img,x_cen, y_cen):
    temp = 0
    return temp

if __name__ == '__main__':

    vis = Visdom(port = 13246,env = 'test')
    all = torch.zeros((2,375,750))
    x = torch.arange(1,376)
    y = torch.arange(1,751)
    for i in range(750):
        all[0,:,i] += x
    for i in range(375):
        all[1,i,:] += y
    x_cen = torch.zeros(3).int()
    x_cen[0] = 195
    x_cen[1] = 3
    x_cen[2] = 7
    y_cen = torch.zeros(3).int()
    y_cen[0] = 400
    y_cen[1] = 10
    y_cen[2] = 9
    sig = 1
    batch = 3
    temp = torch.zeros((batch,375,750))
    for i in range(batch):
        temp[i] = torch.exp(-1 * ((all[0,...]-x_cen[i])**2 + (all[1,...]-y_cen[i])**2)/(2*sig**2))
    print(temp)
    t = torch.zeros((48,375,750))
    print(t[1,x_cen[2],y_cen[2]])
    vis.image(temp[0])
