import torch
from torch import nn
import os
from torchvision import transforms as T
from PIL import Image
from torch.utils.data import Dataset,DataLoader
import numpy as np
import torch.nn.functional as F
from visdom import Visdom
from tqdm import tqdm
from torch import optim
from utils.dataset import seg_patch
from utils.etc import generate_anchors_all_pyramids
from utils.loss import cls_loss, reg_loss
from utils.transbbox import anchor_trans
import random
def network(model, epochs = 100, batch_size = 40, lr = 0.001):
    best_loss = 1000000.0
    vis = Visdom(port = 13246, env = 'RPN1')

    optimizer = optim.Adam(model.parameters(),lr = lr, betas = (0.9, 0.98), weight_decay = 0.0005)

    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.99)

    CE = nn.CrossEntropyLoss()
    smooth_L1 = nn.SmoothL1Loss()
    MSE = nn.MSELoss()

    running_corrects = 0.
    total_loss = 0
    cls, reg = 0.0, 0.0
    div = 0

    p5_anchor = generate_anchors_all_pyramids(scales=[256],ratios=(0.5,1,2),feature_shapes=([5,5]),feature_strides=75,anchor_stride=1).to(0)
    p4_anchor = generate_anchors_all_pyramids(scales=[128],ratios=(0.5,1,2),feature_shapes=([5,5]),feature_strides=75,anchor_stride=1).to(0)
    p3_anchor = generate_anchors_all_pyramids(scales=[64],ratios=(0.5,1,2),feature_shapes=([11,11]),feature_strides=34,anchor_stride=1).to(0)
    p2_anchor = generate_anchors_all_pyramids(scales=[32],ratios=(0.5,1,2),feature_shapes=([23,23]),feature_strides=17,anchor_stride=1).to(0)
    total_anchor = torch.cat((p2_anchor, p3_anchor, p4_anchor, p5_anchor))

    epoch_bar = tqdm(total=epochs,desc = 'epoch')
    for epoch in range(epochs):
        train_data = seg_patch(root = '/home/junkyu/project/tumor_detection/densenet_RPN/data/',transform = T.Compose([
                                    T.RandomRotation(5),
                                    T.RandomVerticalFlip(),
                                    T.RandomHorizontalFlip(),
                                    T.ColorJitter(brightness=random.random(),
                                                           contrast=random.random(),
                                                           saturation=random.random(),
                                                           hue=random.random() / 2),
                                    T.ToTensor(),
                                    T.Normalize(mean = [0.2602,0.2602,0.2602], std = [0.2729,0.2729,0.2729])
                                    ]),mode='train')

        val_data = seg_patch(root = '/home/junkyu/project/tumor_detection/densenet_RPN/data/',transform = T.Compose([
                                    T.ToTensor(),
                                    T.Normalize(mean = [0.2602,0.2602,0.2602], std = [0.2729,0.2729,0.2729])
                                    ]),mode='val')

        train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True, num_workers=2)
        val_loader = DataLoader(val_data, batch_size = batch_size)

        model.train()
        for i,(input, bbox_label, mask) in enumerate(train_loader):
            input, bbox_label = input.to(0), bbox_label.to(0)
            predict_cls, predict_bbox = model(input)

            (anchor_label, iou_value), anchor, predict_anchor, trans_bbox_label = anchor_trans(total_anchor, predict_bbox, bbox_label)
            anchor_label, iou_value = anchor_label.to(0), iou_value.to(0)
            values, idx = torch.sort(iou_value,descending=True)
            loss_cls = cls_loss(predict_cls, anchor_label, idx)
            loss_reg = reg_loss(predict_anchor, trans_bbox_label, idx, anchor_label)
            loss = loss_cls + 0.001* loss_reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            div = div + 1
            total_loss += loss.item()
            cls += loss_cls.item()
            reg += loss_reg.item()
        vis.line(X=[epoch],Y=[total_loss/div],win='loss',name='train',update = 'append',opts=dict(showlegend=True, title='loss'))#, ytickmax=10, ytickmin=0))
        vis.line(X=[epoch],Y=[cls/div],win='cls',name='train',update = 'append',opts=dict(showlegend=True, title='cls'))
        vis.line(X=[epoch],Y=[reg/div],win='reg',name='train',update = 'append',opts=dict(showlegend=True, title='reg'))
        total_loss = 0
        cls, reg = 0.0, 0.0
        div = 0
        with torch.no_grad():
            model.eval()
            running_corrects = 0.
            for i,(input, bbox_label, mask) in enumerate(val_loader):
                input, bbox_label = input.to(0), bbox_label.to(0)
                predict_cls, predict_bbox = model(input)
                # p5_anchor = generate_anchors_all_pyramids(scales=[64,128,256],ratios=(0.5,1,2),feature_shapes=([11,11]),feature_strides=34,anchor_stride=1).to(0)
                (anchor_label, iou_value), anchor, predict_anchor, trans_bbox_label = anchor_trans(total_anchor, predict_bbox, bbox_label)
                anchor_label, iou_value = anchor_label.to(0), iou_value.to(0)
                values, idx = torch.sort(iou_value,descending=True)
                loss_cls = cls_loss(predict_cls, anchor_label, idx)
                loss_reg = reg_loss(predict_anchor, trans_bbox_label, idx, anchor_label)
                loss = loss_cls + 0.001* loss_reg
                div = div + 1
                total_loss += loss.item()
                cls += loss_cls.item()
                reg += loss_reg.item()

            vis.line(X=[epoch], Y=[optimizer.param_groups[0]['lr']], win='lr', name='lr', update='append', opts=dict(showlegend=True, title='learning_rate'))
            if best_loss > total_loss:
                best_loss = total_loss
                torch.save(model.module.state_dict(),'./model/RPN3.pth')
                print(epoch , best_loss)
            vis.line(X=[epoch],Y=[total_loss/div],win='loss',name='val',update = 'append',opts=dict(showlegend=True, title='loss'))#, ytickmax=10, ytickmin=0))
            vis.line(X=[epoch],Y=[cls/div],win='cls',name='val',update = 'append',opts=dict(showlegend=True, title='cls'))
            vis.line(X=[epoch],Y=[reg/div],win='reg',name='val',update = 'append',opts=dict(showlegend=True, title='reg'))
            total_loss = 0
            cls, reg = 0.0, 0.0
            div = 0
            running_corrects = 0
            scheduler.step()
            epoch_bar.update()
    epoch_bar.close()
