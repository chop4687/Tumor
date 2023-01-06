import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict
from torch import Tensor
from torch.jit.annotations import List
import numpy as np
from utils.etc import heatmap
from torchvision import transforms as T

class DenseNet(nn.Module):

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=2, memory_efficient=False):

        super(DenseNet, self).__init__()

        self.one_conv = nn.Conv2d(1024,64,kernel_size = 1,stride = 1)
        self.global_att_conv = nn.Conv2d(512,512,kernel_size = 3,padding=1)
        self.local_one_conv = nn.Conv2d(1536,512,kernel_size = 1,stride = 1)
        self.att_conv1 = nn.Sequential(OrderedDict([
                            ('conv0',nn.Conv2d(128,256,kernel_size = 3,padding=1)),
                            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2)),
                            ('norm0', nn.BatchNorm2d(256)),
                            ('relu0', nn.ReLU(inplace=True)),
                            ('conv1',nn.Conv2d(256,512,kernel_size = 3,padding=1)),
                            ('pool1', nn.MaxPool2d(kernel_size=3, stride=2)),
                            ('norm1', nn.BatchNorm2d(512)),
                            ('relu1', nn.ReLU(inplace=True)),
                    ]))
        self.att_conv2 = nn.Sequential(OrderedDict([
                            ('conv0',nn.Conv2d(256,512,kernel_size = 3,padding=1)),
                            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2)),
                            ('norm0', nn.BatchNorm2d(512)),
                            ('relu0', nn.ReLU(inplace=True)),
                    ]))
        self.att_conv3 = nn.Conv2d(512,512,kernel_size = 3,padding=1)
        self.img_conv = nn.Sequential(OrderedDict([
            #('pool0', nn.MaxPool2d(kernel_size=3, stride=2,padding=1)),
            ('conv0', nn.Conv2d(512,256,kernel_size=3,padding=1)),
            ('norm0', nn.BatchNorm2d(256)),
            ('relu0', nn.ReLU(inplace=True)),
            ('conv1', nn.Conv2d(256,256,kernel_size=3,padding=1)),
            ('norm1', nn.BatchNorm2d(256)),
            ('relu1', nn.ReLU(inplace=True))
        ]))
        self.att_fc1 = nn.Linear(256,4)
        self.att_fc2 = nn.Linear(256,2)
        # self.att_fc1 = nn.Linear(256,128)
        # self.att_fc2 = nn.Linear(128,4)

        self.local_conv = nn.Sequential(OrderedDict([
        ('conv0',nn.Conv2d(256,256,kernel_size=1)),
        ('norm0',nn.BatchNorm2d(256)),
        ('relu0',nn.ReLU(inplace=True)),
        ]))




        # self.reduce_ch1 = nn.Conv2d(768,256,kernel_size=1)
        # self.reduce_ch2 = nn.Conv2d(768,256,kernel_size=1)
    def forward(self, x, model):
        # from visdom import Visdom
        # vis = Visdom(port = 13246, env='test')
        cls, bbox, temp_c2, temp_c3, temp_c4, temp_c5, heat = heatmap(x, model)
        #sigmoid
        # for i in range(8):
        #     vis.image(heat[i,...]*100)

        heat = T.Normalize(heat.mean(),heat.std())(heat)
        sig_heat = nn.Sigmoid()(heat).half().to(0)
        sig_heat = torch.nn.functional.interpolate(sig_heat.unsqueeze(1), size = 11)
        local_feature1 = self.att_conv1(temp_c2)
        local_feature2 = self.att_conv2(temp_c3)
        local_feature3 = self.att_conv3(temp_c4)

        batch, w, h = int(temp_c5.shape[0]/8), temp_c5.shape[2], temp_c5.shape[3]
        local_feature1 = local_feature1.view(batch, 8, -1, w, h)
        local_feature2 = local_feature2.view(batch, 8, -1, w, h)
        local_feature3 = local_feature3.view(batch, 8, -1, w, h)


        temp_c5 = temp_c5 * sig_heat + temp_c5
        global_feature = self.one_conv(temp_c5)

        flat = global_feature.view(batch, -1, w, h)
        global_feature = self.global_att_conv(flat)
        _, channel, w, h = global_feature.shape
        #global feature 종료

        temp1 = torch.matmul(local_feature1,global_feature.unsqueeze(1))
        temp2 = torch.matmul(local_feature2,global_feature.unsqueeze(1))
        temp3 = torch.matmul(local_feature3,global_feature.unsqueeze(1))
        temp = torch.cat((temp1,temp2,temp3),dim=2)

        temp = temp.transpose(2,1)

        patch_temp1 = torch.cat((temp[:,:,0,...],temp[:,:,1,...],temp[:,:,2,...],temp[:,:,3,...]),dim=3)
        patch_temp2 = torch.cat((temp[:,:,4,...],temp[:,:,5,...],temp[:,:,6,...],temp[:,:,7,...]),dim=3)
        patch_temp = torch.cat((patch_temp1,patch_temp2),dim=2)
        patch_temp = self.local_one_conv(patch_temp)
        ############################################
        # local = F.interpolate(local,size=(22,44))
        # local = self.local_conv(local)
        # patch_temp = torch.cat((patch_temp,local),dim=1)
        ###########################################
        patch_temp = self.img_conv(patch_temp)
        img_class = F.adaptive_avg_pool2d(patch_temp,(1,1))
        img_class = torch.flatten(img_class, 1)
        img_hard_class = self.att_fc1(img_class)
        img_mix_class = self.att_fc2(img_class)

        return img_hard_class, img_mix_class, patch_temp#, img_class


def _load_state_dict(model, model_url, progress):
    # '.'s are no longer allowed in module names, but previous _DenseLayer
    # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
    # They are also in the checkpoints in model_urls. This pattern is used
    # to find such keys.
    pattern = re.compile(
        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')

    state_dict = load_state_dict_from_url(model_url, progress=progress)
    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    model.load_state_dict(state_dict)


def _densenet(arch, growth_rate, block_config, num_init_features, pretrained, progress,
              **kwargs):
    model = DenseNet(growth_rate, block_config, num_init_features, **kwargs)
    if pretrained:
        _load_state_dict(model, model_urls[arch], progress)
    return model


def densenet121(pretrained=False, progress=True, **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """
    return _densenet('densenet121', 32, (6, 12, 24, 16), 64, pretrained, progress,
                     **kwargs)
