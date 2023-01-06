import os
from torch.nn.parallel.data_parallel import DataParallel
from network import network
import torch
from torch import nn
from utils.network.rpn import densenet121
from torchvision import models

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '5,6,7'
    backbone_model = models.densenet121(pretrained=True)
    model = densenet121()
    model.first_conv.conv0 = backbone_model.features.conv0
    model.first_conv.norm0 = backbone_model.features.norm0
    model.first_conv.relu0 = backbone_model.features.relu0
    model.first_conv.pool0 = backbone_model.features.pool0

    model.block1[0] = backbone_model.features.denseblock1
    model.block1[1] = backbone_model.features.transition1

    model.block2[0] = backbone_model.features.denseblock2
    model.block2[1] = backbone_model.features.transition2

    model.block3[0] = backbone_model.features.denseblock3
    model.block3[1] = backbone_model.features.transition3

    model.block4[0] = backbone_model.features.denseblock4


    #model.classifier = nn.Linear(model.classifier.in_features, 2)
    # for p in model.first_conv.parameters():
    #     p.requires_grad = False
    #
    # for p in model.block1.parameters():
    #     p.requires_grad = False
    #
    # for p in model.block2.parameters():
    #     p.requires_grad = False
    #
    # for p in model.block3.parameters():
    #     p.requires_grad = False
    #
    # for p in model.block4.parameters():
    #     p.requires_grad = False

    model = DataParallel(model).to(0)
    network(model,epochs = 200, batch_size = 48, lr = 0.0001)
