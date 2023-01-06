import sys
import os
import numpy as np
import torch
from torch import optim
from torch import nn
from torch.nn.parallel.data_parallel import DataParallel
from torchvision import transforms as T
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from visdom import Visdom
from tqdm import tqdm
from utils.transform import Resize, RandomHorizontalFlip, RandomCrop, RandomRotation
from utils.dataset import pretrain_sample
import torch.nn.functional as F


class crop(object):
    def __init__(self,size=1):
        self.size = size
    def __call__(self,tup):
        img,target = tup
        img = np.array(img)
        x = img.shape[0]
        y = img.shape[1]
        img[int(x*0.85):x,int(y*0.85):y] = 0
        img[int(x*0.85):x,0:int(y*0.15)] = 0
        return Image.fromarray(img),target


def train_net(net1,
              net2,
              epochs = 5,
              batch_size = 1,
              lr = 0.0001,
              CV = 1
              ):
    vis = Visdom(port=13246,env='panorama' + str(CV))
    vis_image = Visdom(port=13246,env='panorama-image' + str(CV))
    train_set = pretrain_sample(root = '/home/junkyu/data/1700_tumor_annotation',transform=T.Compose([RandomCrop(),
                                                                                    RandomRotation(5),
                                                                                    RandomHorizontalFlip(),
                                                                                    Resize((512,1024)),
                                                                                    T.Lambda(lambda x: (T.ToTensor()(x[0]), T.ToTensor()(x[1])))]),CV = CV)

    validation_set = pretrain_sample(root = '/home/junkyu/data/1700_tumor_annotation',transform=T.Compose([Resize((512,1024)),
                                                                                    T.Lambda(lambda x: (T.ToTensor()(x[0]), T.ToTensor()(x[1])))]), train = 'val', CV = CV)

    train_loader = DataLoader(train_set,batch_size = batch_size,shuffle = True)
    validation_loader = DataLoader(validation_set,batch_size = batch_size,shuffle = False)

    optimizer = optim.Adam([
                            {'params' : net1.parameters()},
                            {'params' : net2.parameters()}
                          ],
                          lr=lr,
                          betas=(0.9, 0.999),
                          weight_decay=0.0005)

    scheduler = optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.99)

    #MSE = nn.MSELoss()
    BCE = nn.BCELoss()

    epoch_bar = tqdm(total=epochs, desc='epoch')
    total_loss = 0
    best_loss = 100
    sum_loss = 0
    for epoch in range(epochs):
        net1.train()
        net2.train()
        for i, (input , label) in enumerate(train_loader):
            input, label = input.to(0), label.to(0)
            x1,x2,x3,x4,x5 = net1(input)
            x1,x2,x3,x4,x5 = x1.to(0),x2.to(0),x3.to(0),x4.to(0),x5.to(0)
            predict1, predict2, predict3 = net2(x1=x1,x2=x2,x3=x3,x4=x4,x5=x5)

            label1 = F.interpolate(label,
                                (predict1.shape[-2], predict1.shape[-1]),
                                mode='bilinear',
                                align_corners=True)

            label2 = F.interpolate(label,
                                (predict2.shape[-2], predict2.shape[-1]),
                                mode='bilinear',
                                align_corners=True)

            loss1 = BCE(predict1,label1)
            loss2 = BCE(predict2,label2)
            loss3 = BCE(predict3,label)

            sum_loss = loss1 + loss2 + loss3

            total_loss += sum_loss.item()

            optimizer.zero_grad()
            sum_loss.backward()
            optimizer.step()

        vis.line(X=[epoch],Y=[total_loss/len(train_loader)],win='loss',name = 'train',update = 'append',opts=dict(showlegend=True, title='loss'))

        total_loss = 0

        net1.eval()
        net2.eval()
        for i, (input , label) in enumerate(validation_loader):
            input, label = input.to(0), label.to(0)
            x1,x2,x3,x4,x5 = net1(input)
            x1,x2,x3,x4,x5 = x1.to(0),x2.to(0),x3.to(0),x4.to(0),x5.to(0)

            predict1, predict2, predict3 = net2(x1=x1,x2=x2,x3=x3,x4=x4,x5=x5)

            label1 = F.interpolate(label,
                                (predict1.shape[-2], predict1.shape[-1]),
                                mode='bilinear',
                                align_corners=True)

            label2 = F.interpolate(label,
                                (predict2.shape[-2], predict2.shape[-1]),
                                mode='bilinear',
                                align_corners=True)

            loss1 = BCE(predict1,label1)
            loss2 = BCE(predict2,label2)
            loss3 = BCE(predict3,label)

            sum_loss = loss1 + loss2 + loss3

            total_loss += sum_loss.item()

        vis.line(X=[epoch],Y=[total_loss/len(validation_loader)],win='loss',name='val',update = 'append',opts=dict(showlegend=True, title='loss'))
        if epoch % 10 == 0:
            vis_image.image(input[0,...])
            vis_image.image(predict1[0,...])
            vis_image.image(predict2[0,...])
            vis_image.image(predict3[0,...])
            vis_image.image(label[0,...])

        if best_loss > total_loss:
            best_loss = total_loss
            torch.save(net1.module.state_dict(), "./model/Unet_encoder_"+str(CV))
            torch.save(net2.module.state_dict(), "./model/Unet_decoder_"+str(CV))
        total_loss = 0
        scheduler.step()
        epoch_bar.update()
    epoch_bar.close()
