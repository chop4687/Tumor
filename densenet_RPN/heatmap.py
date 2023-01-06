import torch
from torch import nn
import os
from os import path
from torchvision import models

from utils.dataset import seg_patch
from utils.etc import generate_anchors_all_pyramids
from utils.loss import cls_loss, reg_loss
from utils.transbbox import anchor_trans
from torchvision import transforms as T
import random
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.nn import functional as F
from torch import optim
from visdom import Visdom
from torch.nn.parallel.data_parallel import DataParallel
from PIL import ImageDraw

def scale(ten):
    return (ten - ten.min()) / (ten.max() - ten.min())


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    batch_size = 1

    # p5_anchor = generate_anchors_all_pyramids(scales=[256],ratios=(0.5,1,2),feature_shapes=([11,11]),feature_strides=34,anchor_stride=1).to(0)
    # p4_anchor = generate_anchors_all_pyramids(scales=[128],ratios=(0.5,1,2),feature_shapes=([11,11]),feature_strides=34,anchor_stride=1).to(0)
    # p3_anchor = generate_anchors_all_pyramids(scales=[64],ratios=(0.5,1,2),feature_shapes=([23,23]),feature_strides=16,anchor_stride=1).to(0)
    # p2_anchor = generate_anchors_all_pyramids(scales=[32],ratios=(0.5,1,2),feature_shapes=([47,47]),feature_strides=8,anchor_stride=1).to(0)

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
                                ]),mode='test')
    train_loader = DataLoader(train_data, batch_size = batch_size, shuffle=True, num_workers = 4)

    #optimizer = optim.Adam(model.parameters(),lr = 0.001, betas = (0.9, 0.98), weight_decay = 0.0005)

    vis = Visdom(port = 13246, env = 'test')
    count = 0
    count_not = 0
    div = 0
    for i,(input, bbox_label, mask) in enumerate(train_loader):
        input, bbox_label = input.to(0), bbox_label.to(0)
        print(bbox_label.shape)
        vis.image(input[0,...])
        vis.image(mask)
        from utils.network.rpn import densenet121
        model = densenet121()
        model.load_state_dict(torch.load('/home/junkyu/project/tumor_detection/densenet_RPN/model/RPN3.pth'))
        model = DataParallel(model).to(0)
        model.eval()
        predict_cls, predict_bbox = model(input)
        p5_anchor = generate_anchors_all_pyramids(scales=[256],ratios=(0.5,1,2),feature_shapes=([5,5]),feature_strides=75,anchor_stride=1).to(0)
        p4_anchor = generate_anchors_all_pyramids(scales=[128],ratios=(0.5,1,2),feature_shapes=([5,5]),feature_strides=75,anchor_stride=1).to(0)
        p3_anchor = generate_anchors_all_pyramids(scales=[64],ratios=(0.5,1,2),feature_shapes=([11,11]),feature_strides=34,anchor_stride=1).to(0)
        p2_anchor = generate_anchors_all_pyramids(scales=[32],ratios=(0.5,1,2),feature_shapes=([23,23]),feature_strides=17,anchor_stride=1).to(0)

        total_anchor = torch.cat((p2_anchor, p3_anchor, p4_anchor, p5_anchor))

        (anchor_label, iou_value), anchor, predict_anchor, trans_bbox_label = anchor_trans(total_anchor, predict_bbox, bbox_label)
        anchor_label, iou_value = anchor_label.to(0), iou_value.to(0)
        arg_cls = nn.Softmax(dim=2)(predict_cls)
        value,idx = torch.sort(arg_cls[...,1],descending=True)
        pred = predict_anchor[0,idx[0,0:100]]
        black = torch.zeros((375,375))
        kk = T.ToPILImage()(input[0,...].cpu())
        kk1 = ImageDraw.Draw(kk)

        if arg_cls[0,idx[0,0],1] < 0.4:
            x_min, y_min, w, h = pred[0,:].int()
            black[x_min:(x_min + w),y_min:(y_min+h)] += arg_cls[0,idx[0,0],1]
            shape = [(pred[0][1],pred[0][0]),(pred[0][1]+pred[0][3],pred[0][0]+pred[0][2])]
            kk1.rectangle(shape,outline='red',width=2)
            print(arg_cls[0,idx[0,0],1])

        for k in range(30):
            if arg_cls[0,idx[0,k],1] >= 0.4:
                x_min, y_min, w, h = pred[k,:].int()
                black[x_min:(x_min + w),y_min:(y_min+h)] += arg_cls[0,idx[0,k],1]
                shape = [(pred[k][1],pred[k][0]),(pred[k][1]+pred[k][3],pred[k][0]+pred[k][2])]
                kk1.rectangle(shape,outline='red',width=2)
        black = scale(black)
        temp = torch.where(black == 1)
        # if len(temp[0]) == 0:
        #     count_not += 1
        #     vis.image(input[0,...])
        #     vis.image(T.ToTensor()(kk))
        #     vis.image(mask)
        #     vis.image(black)
        x_min,y_min,x_max,y_max = temp[0].min(), temp[1].min(), temp[0].max(),temp[1].max()
        w = x_max - x_min
        h = y_max - y_min
        ctr_x = (x_min+(w//2)).int()
        ctr_y = (y_min+(h//2)).int()
        #print(mask[0,0,ctr_x,ctr_y])
        if mask[0,0,ctr_x,ctr_y] == 0:
            count += 1
            # vis.image(input[0,...])
            # vis.image(T.ToTensor()(kk))
            # vis.image(mask)
            # vis.image(black)
            # exit()
        # if arg_cls[0,idx[0,0],1] < 0.4:
        #     mask[0,0,(ctr_x-2):(ctr_x+2),(ctr_y-2):(ctr_y+2)] = 0
        #     vis.image(input[0,...])
        #     vis.image(T.ToTensor()(kk))
        #     vis.image(mask)
        #     vis.image(black)
        mask[0,0,(ctr_x-2):(ctr_x+2),(ctr_y-2):(ctr_y+2)] = 0
        #vis.image(input[0,...])
        vis.image(T.ToTensor()(kk))
        #vis.image(mask)
        vis.image(black)
        div += 1
        exit()
    print(count_not,count, div)
