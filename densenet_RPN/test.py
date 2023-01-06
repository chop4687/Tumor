import torch
from torch import nn
import os
from os import path
from torchvision import models

from utils.etc import generate_anchors_all_pyramids
from utils.trans_bbox import anchor_trans
from utils.loss import cls_loss, reg_loss

from model_RPN import densenet121
from dataset import balence_patch, already_patch,seg_patch
from torchvision import transforms as T
import random
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.nn import functional as F
from torch import optim
from visdom import Visdom
from torch.nn.parallel.data_parallel import DataParallel
from PIL import ImageDraw

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    batch_size = 1
    p5_anchor = generate_anchors_all_pyramids(scales=[256],ratios=(0.5,1,2),feature_shapes=([5,5]),feature_strides=75,anchor_stride=1).to(0)
    p4_anchor = generate_anchors_all_pyramids(scales=[128],ratios=(0.5,1,2),feature_shapes=([5,5]),feature_strides=75,anchor_stride=1).to(0)
    p3_anchor = generate_anchors_all_pyramids(scales=[64],ratios=(0.5,1,2),feature_shapes=([11,11]),feature_strides=34,anchor_stride=1).to(0)
    p2_anchor = generate_anchors_all_pyramids(scales=[32],ratios=(0.5,1,2),feature_shapes=([23,23]),feature_strides=17,anchor_stride=1).to(0)
    total_anchor = torch.cat((p2_anchor, p3_anchor, p4_anchor, p5_anchor))
    train_data = seg_patch(root = '/home/junkyu/project/tumor_detection/densenet_RPN/data/',transform = T.Compose([
                                # T.RandomRotation(5),
                                # T.RandomVerticalFlip(),
                                # T.RandomHorizontalFlip(),
                                # T.ColorJitter(brightness=random.random(),
                                #                        contrast=random.random(),
                                #                        saturation=random.random(),
                                #                        hue=random.random() / 2),
                                T.ToTensor(),
                                #T.Normalize(mean = [0.2602,0.2602,0.2602], std = [0.2729,0.2729,0.2729])
                                ]),mode='val')
    train_loader = DataLoader(train_data, batch_size = batch_size, shuffle=False, num_workers = 4)
    MSE = nn.MSELoss()

    model = densenet121()
    #model.load_state_dict(torch.load('/home/junkyu/project/re_tumor/RPN/model/dense_rpn6.pth'))
    model = DataParallel(model).to(0)
    optimizer = optim.Adam(model.parameters(),lr = 0.001, betas = (0.9, 0.98), weight_decay = 0.0005)
    scales = [32, 64, 128]
    ratios = [0.5, 1, 2]
    model.eval()
    loss_cls = 0.0
    vis = Visdom(port = 13246, env = 'test')
    for i,(input, bbox_label, mask) in enumerate(train_loader):
        input, bbox_label = input.to(0), bbox_label.to(0)
        predict_cls, predict_bbox = model(input)
        (anchor_label, iou_value), anchor, predict_anchor, trans_bbox_label = anchor_trans(total_anchor, predict_bbox, bbox_label)
        anchor_label, iou_value = anchor_label.to(0), iou_value.to(0)
        ttt = nn.Softmax(dim=2)(predict_cls)
        values, idx = torch.sort(iou_value,descending=True)
        print(anchor_label[0,idx[0,0:40]])
        # v,id = torch.sort(ttt[...,1],descending=True)
        # a = predict_anchor[0,id[0,0:100],:]
        # vis.image(input[0,...])
        # kk = T.ToPILImage()(input[0,...].cpu())
        # kk1 = ImageDraw.Draw(kk)
        # for b in range(100):
        #     if torch.argmax(predict_cls[:,id[0,0:100],:],dim=2)[0,b] == 1:
        #         shape = [(a[b][1],a[b][0]),(a[b][1]+a[b][3],a[b][0]+a[b][2])]
        #         kk1.rectangle(shape,outline='red',width=2)
        # vis.image(T.ToTensor()(kk))
        # vis.image(mask)
        # exit()
        # loss_cls = cls_loss(predict_cls, anchor_label, idx)
        # loss_reg = reg_loss(predict_anchor, trans_bbox_label, idx, anchor_label)
        # print(loss_cls)
        # print(loss_reg)
        # loss = loss_cls + loss_reg
        # loss.backward()
