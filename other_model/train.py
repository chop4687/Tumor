import os
import torch
from torchvision.models import densenet121
from network import network
from torch.nn.parallel.data_parallel import DataParallel
from torch import nn
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3,4,5,6,7,8'
    model = densenet121(pretrained=True)
    model.classifier = nn.Linear(1024,4)
    model = DataParallel(model).to(0)
    network(model,epochs = 200, batch_size = 24, lr = 0.0001)
