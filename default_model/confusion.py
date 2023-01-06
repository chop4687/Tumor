from visdom import Visdom
from tqdm import tqdm
from torch import nn, optim
from torchvision import transforms as T
from dataset import binary_classification
from torch.utils.data import DataLoader
import torch
import os
import random
from torchvision.models import densenet121
from torch.nn.parallel.data_parallel import DataParallel
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    model = densenet121(pretrained=True)
    model.classifier = nn.Linear(1024,4)
    vis = Visdom(port = 13246,env='test')
    testset = binary_classification(root = '/home/junkyu/project/tumor_detection/default_model/data/',
                                transform = T.Compose([
                                                        T.ToTensor(),
                                                        T.Normalize(mean = [0.2602,0.2602,0.2602], std = [0.2729,0.2729,0.2729])
                                ]),mode = True)
    test_loader = DataLoader(testset,shuffle=False,batch_size=1,num_workers=2)

    model.load_state_dict(torch.load('/home/junkyu/project/tumor_detection/default_model/four.pth'))
    model = DataParallel(model).to(0)
    model.eval()
    ff,div = 0,0
    zz,zo,oz,oo = 0,0,0,0
    conf = torch.zeros((4,4))
    with torch.no_grad():
        for i,(input,target) in enumerate(test_loader):
            input, target = input.to(0), target.to(0)
            predict = model(input)
            #print(predict)
            pred = torch.argmax(predict,1)
            conf[target,pred] += 1
            if target == pred:
                ff += 1
            div += 1
    print(conf)
    print(ff,div)
