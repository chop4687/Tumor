import os
from torch.nn.parallel.data_parallel import DataParallel
from network import network
import torch
import torch.nn as nn
from utils.network.model import densenet121
from utils.network.unet import UNet
from utils.network.rpn import densenet121_RPN
import torchvision
if __name__ == '__main__':
    for cv in range(5):
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'
        model = densenet121()
        densenet = densenet121_RPN()
        # temp_model = torchvision.models.densenet121()
        # temp_model.classifier = nn.Linear(1024,3)
        # temp_model.load_state_dict(torch.load('/home/junkyu/project/tumor_detection/default_model/three.pth'))
        # densenet.first_conv.conv0 = temp_model.features.conv0
        # densenet.first_conv.norm0 = temp_model.features.norm0
        # densenet.first_conv.relu0 = temp_model.features.relu0
        # densenet.first_conv.pool0 = temp_model.features.pool0
        #
        # densenet.block1[0] = temp_model.features.denseblock1
        # densenet.block1[1] = temp_model.features.transition1
        #
        # densenet.block2[0] = temp_model.features.denseblock2
        # densenet.block2[1] = temp_model.features.transition2
        #
        # densenet.block3[0] = temp_model.features.denseblock3
        # densenet.block3[1] = temp_model.features.transition3
        #
        # densenet.block4[0] = temp_model.features.denseblock4
        densenet.load_state_dict(torch.load('/home/junkyu/project/tumor_detection/densenet_RPN/model/RPN3.pth'))
        for p in densenet.parameters():
            p.requires_grad = False
        densenet.eval()
        #model.load_state_dict(torch.load('/home/junkyu/project/re_tumor/model/dense.pth'))
        model_unet = UNet(n_channels = 256, n_classes = 1)
        # for param in model.parameters():
        #     param.requires_grad = False
        #
        # for p in model.att_conv1.parameters():
        #     p.requires_grad = True
        #
        # for p in model.att_conv2.parameters():
        #     p.requires_grad = True
        #
        # for p in model.att_conv3.parameters():
        #     p.requires_grad = True
        #
        # for p in model.one_conv.parameters():
        #     p.requires_grad = True
        #
        # for p in model.global_att_conv.parameters():
        #     p.requires_grad = True
        #
        # for p in model.local_one_conv.parameters():
        #     p.requires_grad = True
        #
        # for p in model.img_conv.parameters():
        #     p.requires_grad = True
        #
        # for p in model.att_fc1.parameters():
        #     p.requires_grad = True


        model = DataParallel(model).to(0)
        model_unet = DataParallel(model_unet).to(0)
        densenet = DataParallel(densenet).to(0)
        network(model, model_unet, densenet,epochs=200,batch_size=48,lr=0.0001,cv=cv)
