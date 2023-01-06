import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict
from torch import Tensor
from torch.jit.annotations import List
import numpy as np
import random
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

        # First convolution
        self.first_conv = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))
        self.block1 = nn.Sequential(OrderedDict([]))
        self.block2 = nn.Sequential(OrderedDict([]))
        self.block3 = nn.Sequential(OrderedDict([]))
        self.block4 = nn.Sequential(OrderedDict([]))
        self.maxpool = nn.MaxPool2d(2)
        ########################################fpn part
        # Top layer
        self.toplayer = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Lateral layersf
        self.latlayer1 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0)

        # one one class layers
        self.class_score1 = nn.Conv2d(256,6,kernel_size=1, stride=1,padding=0)
        self.bbox1 = nn.Conv2d(256,12,kernel_size=1, stride=1,padding=0)
        self.class_score2 = nn.Conv2d(256,6,kernel_size=1, stride=1,padding=0)
        self.bbox2 = nn.Conv2d(256,12,kernel_size=1, stride=1,padding=0)
        self.class_score3 = nn.Conv2d(256,6,kernel_size=1, stride=1,padding=0)
        self.bbox3 = nn.Conv2d(256,12,kernel_size=1, stride=1,padding=0)
        self.class_score4 = nn.Conv2d(256,6,kernel_size=1, stride=1,padding=0)
        self.bbox4 = nn.Conv2d(256,12,kernel_size=1, stride=1,padding=0)
        ##################################################

        ##################################################
        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient
            )
            if i == 0:

                self.block1.add_module('denseblock%d' % (i + 1), block)
                num_features = num_features + num_layers * growth_rate
                if i != len(block_config) - 1:
                    trans = _Transition(num_input_features=num_features,
                                        num_output_features=num_features // 2)
                    self.block1.add_module('transition%d' % (i + 1), trans)
                    num_features = num_features // 2

            elif i == 1:
                self.block2.add_module('denseblock%d' % (i + 1), block)
                num_features = num_features + num_layers * growth_rate
                if i != len(block_config) - 1:
                    trans = _Transition(num_input_features=num_features,
                                        num_output_features=num_features // 2)
                    self.block2.add_module('transition%d' % (i + 1), trans)
                    num_features = num_features // 2
            elif i == 2:
                self.block3.add_module('denseblock%d' % (i + 1), block)
                num_features = num_features + num_layers * growth_rate
                if i != len(block_config) - 1:
                    trans = _Transition(num_input_features=num_features,
                                        num_output_features=num_features // 2)
                    self.block3.add_module('transition%d' % (i + 1), trans)
                    num_features = num_features // 2
            else:
                self.block4.add_module('denseblock%d' % (i + 1), block)
                num_features = num_features + num_layers * growth_rate
                if i != len(block_config) - 1:
                    trans = _Transition(num_input_features=num_features,
                                        num_output_features=num_features // 2)
                    self.block4.add_module('transition%d' % (i + 1), trans)
                    num_features = num_features // 2

        # Final batch norm
        self.last_batch_norm = nn.Sequential(OrderedDict([]))
        self.last_batch_norm.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, 2)
        #self.classifier = nn.Linear(4096, 2)
        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def _upsample_add(self, x, y):
        _,_,H,W = y.size()
        return F.interpolate(x, size=(H,W), mode='bilinear', align_corners=True) + y

    def forward(self, x):
        batch = x.shape[0]
        out = self.first_conv(x)
        c2 = self.block1(out)
        c3 = self.block2(c2)
        c4 = self.block3(c3)
        c5 = self.block4(c4)

        c2 = self.maxpool(c2)
        c3 = self.maxpool(c3)
        c4 = self.maxpool(c4)
        c5 = self.maxpool(c5)

        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5,self.latlayer1(c4))
        p3 = self._upsample_add(p4,self.latlayer2(c3))
        p2 = self._upsample_add(p3,self.latlayer3(c2))

        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)

        p2_cls = self.class_score1(p2)
        p2_cls = p2_cls.reshape(batch,3,2,p2.shape[-1],p2.shape[-1])
        # p2_cls = nn.Softmax(dim=2)(p2_cls)
        p2_cls = p2_cls.permute(0,1,3,4,2)
        p2_cls = p2_cls.reshape(batch,-1,2)

        p2_bbox = self.bbox1(p2)
        p2_bbox = p2_bbox.reshape(batch,3,4,p2.shape[-1],p2.shape[-1])
        p2_bbox = p2_bbox.permute(0,1,3,4,2)
        p2_bbox = p2_bbox.reshape(batch,-1,4)

        p3_cls = self.class_score2(p3)
        p3_cls = p3_cls.reshape(batch,3,2,p3.shape[-1],p3.shape[-1])
        # p3_cls = nn.Softmax(dim=2)(p3_cls)
        p3_cls = p3_cls.permute(0,1,3,4,2)
        p3_cls = p3_cls.reshape(batch,-1,2)

        p3_bbox = self.bbox2(p3)
        p3_bbox = p3_bbox.reshape(batch,3,4,p3.shape[-1],p3.shape[-1])
        p3_bbox = p3_bbox.permute(0,1,3,4,2)
        p3_bbox = p3_bbox.reshape(batch,-1,4)

        p4_cls = self.class_score3(p4)
        p4_cls = p4_cls.reshape(batch,3,2,p4.shape[-1],p4.shape[-1])
        # p4_cls = nn.Softmax(dim=2)(p4_cls)
        p4_cls = p4_cls.permute(0,1,3,4,2)
        p4_cls = p4_cls.reshape(batch,-1,2)

        p4_bbox = self.bbox3(p4)
        p4_bbox = p4_bbox.reshape(batch,3,4,p4.shape[-1],p4.shape[-1])
        p4_bbox = p4_bbox.permute(0,1,3,4,2)
        p4_bbox = p4_bbox.reshape(batch,-1,4)

        p5_cls = self.class_score4(p5)
        p5_cls = p5_cls.reshape(batch,3,2,p5.shape[-1],p5.shape[-1])
        # p5_cls = nn.Softmax(dim=2)(p5_cls)
        p5_cls = p5_cls.permute(0,1,3,4,2)
        p5_cls = p5_cls.reshape(batch,-1,2)

        p5_bbox = self.bbox4(p5)
        p5_bbox = p5_bbox.reshape(batch,3,4,p5.shape[-1],p5.shape[-1])
        p5_bbox = p5_bbox.permute(0,1,3,4,2)
        p5_bbox = p5_bbox.reshape(batch,-1,4)
        cls = torch.cat((p2_cls,p3_cls,p4_cls,p5_cls),dim=1)
        bbox = torch.cat((p2_bbox,p3_bbox,p4_bbox,p5_bbox),dim=1)
        # cls = (p3_cls,p4_cls,p5_cls)
        # bbox = (p3_bbox,p4_bbox,p5_bbox)
        return cls, bbox

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
