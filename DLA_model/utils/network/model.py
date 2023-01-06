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

def scale(ten):
    return (ten - ten.min()) / (ten.max() - ten.min())
class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False)),
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs):
        # type: (List[Tensor]) -> Tensor
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        return bottleneck_output

    # todo: rewrite when torchscript supports any
    def any_requires_grad(self, input):
        # type: (List[Tensor]) -> bool
        for tensor in input:
            if tensor.requires_grad:
                return True
        return False

    @torch.jit.unused  # noqa: T484
    def call_checkpoint_bottleneck(self, input):
        # type: (List[Tensor]) -> Tensor
        def closure(*inputs):
            return self.bn_function(*inputs)

        return cp.checkpoint(closure, input)

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input):
        # type: (List[Tensor]) -> (Tensor)
        pass

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input):
        # type: (Tensor) -> (Tensor)
        pass

    # torchscript does not yet support *args, so we overload method
    # allowing it to take either a List[Tensor] or single Tensor
    def forward(self, input):  # noqa: F811
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception("Memory Efficient not supported in JIT")

            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bn_function(prev_features)

        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features

def scale(ten):
    return (ten - ten.min()) / (ten.max() - ten.min())

class _DenseBlock(nn.ModuleDict):
    _version = 2

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, memory_efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))

class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """

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
        # self.att_fc1 = nn.Linear(256,128)
        # self.att_fc2 = nn.Linear(128,4)

        self.local_conv = nn.Sequential(OrderedDict([
        ('conv0',nn.Conv2d(256,1,kernel_size=1,stride=1)),
        ('norm0',nn.BatchNorm2d(1)),
        ('relu0',nn.ReLU(inplace=True))
        ]))

        self.conv1 = nn.Conv2d(256,256,kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(256,256,kernel_size=3,padding=1)
        self.conv3 = nn.Conv2d(256,256,kernel_size=3,padding=1)
        self.conv4 = nn.Conv2d(256,256,kernel_size=3,padding=1)
        self.conv5 = nn.Conv2d(256,256,kernel_size=3,padding=1)
        self.conv6 = nn.Conv2d(256,256,kernel_size=3,padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.relu = nn.ReLU(inplace=True)
        self.last_one = nn.Sequential(OrderedDict([
        ('conv0',nn.Conv2d(256,1,kernel_size=1)),
        ('norm0',nn.BatchNorm2d(1)),
        ('relu0',nn.ReLU(inplace=True))
        ]))
        self.reduce_ch1 = nn.Sequential(OrderedDict([
        ('conv0',nn.Conv2d(768,256,kernel_size=1)),
        ('norm0',nn.BatchNorm2d(256)),
        ('relu0',nn.ReLU(inplace=True))
        ]))
        self.reduce_ch2 = nn.Sequential(OrderedDict([
        ('conv0',nn.Conv2d(768,256,kernel_size=1)),
        ('norm0',nn.BatchNorm2d(256)),
        ('relu0',nn.ReLU(inplace=True))
        ]))
        # self.reduce_ch1 = nn.Conv2d(768,256,kernel_size=1)
        # self.reduce_ch2 = nn.Conv2d(768,256,kernel_size=1)
    def forward(self, big_x, x, model):
        # from visdom import Visdom
        # vis = Visdom(port = 13246, env='test')
        cls, bbox, temp_c2, temp_c3, temp_c4, temp_c5, heat = heatmap(x, model)
        #sigmoid
        # for i in range(8):
        #     vis.image(heat[i,...]*100)

        heat = T.Normalize(heat.mean(),heat.std())(heat)
        sig_heat = nn.Sigmoid()(heat).to(0)
        sig_heat = torch.nn.functional.interpolate(sig_heat.unsqueeze(1), size = 11)
        local_feature1 = self.att_conv1(temp_c2)
        local_feature2 = self.att_conv2(temp_c3)
        local_feature3 = self.att_conv3(temp_c4)

        batch, w, h = int(temp_c5.shape[0]/8), temp_c5.shape[2], temp_c5.shape[3]
        local_feature1 = local_feature1.view(batch, 8, -1, w, h)
        local_feature2 = local_feature2.view(batch, 8, -1, w, h)
        local_feature3 = local_feature3.view(batch, 8, -1, w, h)
        #global feature 만들기
        #여기에 구한 heatmap을 곱한다.

        temp_c5 = temp_c5 * sig_heat + temp_c5
        global_feature = self.one_conv(temp_c5)

        # global_feature = global_feature * sig_heat + global_feature
        #print(torch.unique(global_feature[6,0,...]))
        # vis.image(torch.nn.functional.interpolate(scale(x), size = 300)[6,0,...])
        # vis.image(torch.nn.functional.interpolate(sig_heat,size=300)[6,...])
        # vis.image(torch.nn.functional.interpolate(temp_c5.sum(dim=1).unsqueeze(1), size = 300)[6,...])
        # vis.image(torch.nn.functional.interpolate(kk.sum(dim=1).unsqueeze(1), size = 300)[6,...])
        # exit()
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
        patch_temp = self.img_conv(patch_temp)

        patch_exp = self.conv1(patch_temp)
        patch_temp = self.conv2(patch_exp)
        patch_temp = patch_temp + patch_exp
        patch_temp = self.pool(patch_temp)
        skip_one = patch_temp
        #patch_temp -> Decoder로 보낼꺼임

        patch_exp = self.conv3(patch_temp)
        patch_temp = self.conv4(patch_exp)
        patch_temp = patch_temp + patch_exp
        patch_temp = self.pool(patch_temp)
        skip_two = patch_temp
        up_skip_one = F.relu(F.interpolate(patch_temp, size=(11,22)))

        #patch_temp -> Decoder로 보낼꺼임

        patch_exp = self.conv5(patch_temp)
        patch_temp = self.conv6(patch_exp)
        patch_temp = patch_temp + patch_exp
        patch_temp = self.pool(patch_temp)
        up_skip_two = F.relu(F.interpolate(patch_temp, size=(5,11)))

        img_class = F.adaptive_avg_pool2d(patch_temp,(1,1))
        img_class = torch.flatten(img_class, 1)
        img_class = self.att_fc1(img_class)
        #img_class = self.att_fc2(img_class)
        #patch_temp -> Decoder로 보낼꺼임

        ##decoder 부분 여기서 resize를 하는데 크기가 맞아야 뭐가 가..
        patch_temp = self.up1(patch_temp)
        patch_temp = F.interpolate(patch_temp,size=(5,11))
        patch_temp = torch.cat((patch_temp,skip_two,up_skip_two),dim=1)
        patch_temp = self.reduce_ch1(patch_temp)
        # 5 x 11
        patch_temp = self.up2(patch_temp)
        patch_temp = F.interpolate(patch_temp,size=(11,22))
        patch_temp = torch.cat((patch_temp,skip_one,up_skip_one),dim=1)
        patch_temp = self.reduce_ch2(patch_temp)
        # 11 x 22
        patch_temp = self.up3(patch_temp)
        #patch_local = patch_temp.sum(dim=1).unsqueeze(1)
        patch_local = self.last_one(patch_temp)
        patch_local = F.sigmoid(patch_local)
        # 22 x 44
        return img_class, patch_local


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
