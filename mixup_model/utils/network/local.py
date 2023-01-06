from torch import nn
from collections import OrderedDict
from torch.nn import functional as F
import torch
class DLA_model(nn.Module):
    def __init__(self,num_input_features=64):
        super(DLA_model,self).__init__()

        self.inv = nn.Sequential(OrderedDict([
        ('conv0',nn.Conv2d(3,32,kernel_size=3,padding=1,dilation=3,stride=2)),
        ('norm0',nn.BatchNorm2d(32)),
        ('relu0',nn.ReLU(inplace=True)),
        ]))
        self.down1_conv1 = nn.Sequential(OrderedDict([
        ('conv0',nn.Conv2d(32,64,kernel_size=3,padding=1,stride=1)),
        ('norm0',nn.BatchNorm2d(64)),
        ('relu0',nn.ReLU(inplace=True)),
        ]))
        self.down1_conv2 = nn.Sequential(OrderedDict([
        ('conv0',nn.Conv2d(64,64,kernel_size=3,padding=1,stride=1)),
        ('norm0',nn.BatchNorm2d(64)),
        ('relu0',nn.ReLU(inplace=True))
        ]))

        self.down2_conv1 = nn.Sequential(OrderedDict([
        ('conv0',nn.Conv2d(64,128,kernel_size=3,padding=1,stride=1)),
        ('norm0',nn.BatchNorm2d(128)),
        ('relu0',nn.ReLU(inplace=True)),
        ]))
        self.down2_conv2 = nn.Sequential(OrderedDict([
        ('conv0',nn.Conv2d(128,128,kernel_size=3,padding=1,stride=1)),
        ('norm0',nn.BatchNorm2d(128)),
        ('relu0',nn.ReLU(inplace=True))
        ]))

        self.down3_conv1 = nn.Sequential(OrderedDict([
        ('conv0',nn.Conv2d(128,256,kernel_size=3,padding=1,stride=1)),
        ('norm0',nn.BatchNorm2d(256)),
        ('relu0',nn.ReLU(inplace=True)),
        ]))
        self.down3_conv2 = nn.Sequential(OrderedDict([
        ('conv0',nn.Conv2d(256,256,kernel_size=3,padding=1,stride=1)),
        ('norm0',nn.BatchNorm2d(256)),
        ('relu0',nn.ReLU(inplace=True))
        ]))

        self.trans_conv = nn.Sequential(OrderedDict([
        ('conv0',nn.Conv2d(512,512,kernel_size=3,padding=1)),
        ('norm0',nn.BatchNorm2d(512)),
        ('relu0',nn.ReLU(inplace=True))
        ]))

        self.up1 = nn.Sequential(OrderedDict([
        ('up0',nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)),
        ('conv0',nn.Conv2d(512,128,kernel_size=3,padding=1,stride=1)),
        ('norm0',nn.BatchNorm2d(128)),
        ('relu0',nn.ReLU(inplace=True)),
        ('conv1',nn.Conv2d(128,64,kernel_size=3,padding=1,stride=1)),
        ('norm1',nn.BatchNorm2d(64)),
        ('relu1',nn.ReLU(inplace=True)),
        ]))

        self.up2 = nn.Sequential(OrderedDict([
        ('up0',nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)),
        ('conv0',nn.Conv2d(128,64,kernel_size=3,padding=1,stride=1)),
        ('norm0',nn.BatchNorm2d(64)),
        ('relu0',nn.ReLU(inplace=True)),
        ('conv1',nn.Conv2d(64,64,kernel_size=3,padding=1,stride=1)),
        ('norm1',nn.BatchNorm2d(64)),
        ('relu1',nn.ReLU(inplace=True)),
        ]))

        self.up3 = nn.Sequential(OrderedDict([
        ('up0',nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)),
        ('conv0',nn.Conv2d(64,64,kernel_size=3,padding=1,stride=1)),
        ('norm0',nn.BatchNorm2d(64)),
        ('relu0',nn.ReLU(inplace=True)),
        ('conv1',nn.Conv2d(64,64,kernel_size=3,padding=1,stride=1)),
        ('norm1',nn.BatchNorm2d(64)),
        ('relu1',nn.ReLU(inplace=True)),
        ]))
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.reduce_ch1 = nn.Sequential(OrderedDict([
        ('conv0',nn.Conv2d(448,128,kernel_size=1)),
        ('norm0',nn.BatchNorm2d(128)),
        ('relu0',nn.ReLU(inplace=True))
        ]))

        self.reduce_ch2 = nn.Sequential(OrderedDict([
        ('conv0',nn.Conv2d(256,64,kernel_size=1)),
        ('norm0',nn.BatchNorm2d(64)),
        ('relu0',nn.ReLU(inplace=True))
        ]))
        #############################################

        self.last_one = nn.Conv2d(64,1,kernel_size=1,padding=1)


    def forward(self,patch_temp, cls_feature):
        patch_temp = self.inv(patch_temp)
        patch_exp = self.down1_conv1(patch_temp)
        patch_temp = self.down1_conv2(patch_exp)
        patch_temp = patch_temp + patch_exp
        patch_temp = self.pool(patch_temp)
        skip_connection_one = patch_temp


        patch_exp = self.down2_conv1(patch_temp)
        patch_temp = self.down2_conv2(patch_exp)
        patch_temp = patch_temp + patch_exp
        patch_temp = self.pool(patch_temp)
        skip_connection_two = patch_temp

        up_connection_one = F.relu(self.up(patch_temp))


        patch_exp = self.down3_conv1(patch_temp)
        patch_temp = self.down3_conv2(patch_exp)
        patch_temp = patch_temp + patch_exp
        patch_down = self.pool(patch_temp)

        up_connection_two = F.relu(self.up(patch_down))
        cls_feature = F.interpolate(cls_feature,size=(23,46))
        patch_down = torch.cat((cls_feature,patch_down),dim=1)
        patch_down = self.trans_conv(patch_down)

        patch_temp = self.up1(patch_down)
        skip_connection_two = F.interpolate(skip_connection_two,size=(patch_temp.shape[2],patch_temp.shape[3]))
        patch_temp = torch.cat((patch_temp,skip_connection_two,up_connection_two),dim=1)

        patch_temp = self.reduce_ch1(patch_temp)

        patch_temp = self.up2(patch_temp)
        skip_connection_one = F.interpolate(skip_connection_one,size=(patch_temp.shape[2],patch_temp.shape[3]))
        up_connection_one = F.interpolate(up_connection_one,size=(patch_temp.shape[2],patch_temp.shape[3]))
        patch_temp = torch.cat((patch_temp,skip_connection_one,up_connection_one),dim=1)
        patch_temp = self.reduce_ch2(patch_temp)

        patch_temp = self.up3(patch_temp)
        patch_local = self.last_one(patch_temp)
        patch_local = F.interpolate(patch_local,size = (375,750))
        patch_local = F.sigmoid(patch_local)
        return patch_local
