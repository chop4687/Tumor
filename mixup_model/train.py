import os
from torch.nn.parallel.data_parallel import DataParallel
from network import network
import torch
import torch.nn as nn
from utils.network.model import densenet121
from utils.network.rpn import densenet121_RPN
from utils.network.local import DLA_model
from utils.network.unet import UNet
import torchvision
if __name__ == '__main__':
    for cv in range(4,5):
        os.environ['CUDA_VISIBLE_DEVICES'] = '2,3,4'
        model = densenet121()
        densenet = densenet121_RPN()
        localization = DLA_model()
        #localization = UNet(3,1)
        densenet.load_state_dict(torch.load('/home/junkyu/project/tumor_detection/densenet_RPN/model/RPN3.pth'))
        for p in densenet.parameters():
            p.requires_grad = False
        densenet.eval()
        mix_check = True
        model = model.to(0)
        localization = DataParallel(localization).to(0)
        densenet = densenet.to(0)
        network(model, localization, densenet, epochs=200,batch_size=48,lr=0.0005,cv = cv, mix_check = mix_check)
