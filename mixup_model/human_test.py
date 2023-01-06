import torch
from torch.utils.data import DataLoader
from utils.etc import IoU, conf_matrix, find_center
from utils.dataset import tumor, test_tumor
from utils.transform import *
import numpy as np
from utils.network.model import densenet121
#from utils.network.local import localization_model
from utils.network.local import DLA_model
from visdom import Visdom
import os
from torch.nn.parallel.data_parallel import DataParallel
from utils.network.unet import UNet
from utils.network.rpn import densenet121_RPN
from utils.loss import MultiTaskLoss
from utils.mixup import mixup
import torchvision
import random
import torch.nn.functional as F
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
def scale(ten):
    temp = (ten - ten.min()) / (ten.max() - ten.min())

    return temp

if __name__ == '__main__':
    toto = torch.zeros((4,4))
    pp, pre, tar = [], [], []
    kk = []
    AB = os.listdir('/home/junkyu/project/tumor_detection/human_test/AB')
    NOR = os.listdir('/home/junkyu/project/tumor_detection/human_test/NOR')
    OKC = os.listdir('/home/junkyu/project/tumor_detection/human_test/OKC')
    DC = os.listdir('/home/junkyu/project/tumor_detection/human_test/DC')
    total = AB+NOR+OKC+DC
    for k in range(5):
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'
        vis = Visdom(port = 13246,env='test3')
        testset = test_tumor(root = '/home/junkyu/project/tumor_detection/DLA_model/data200',transform = T.Compose([
                                    ToTensor(),
                                    Normalize(mean = [0.2602], std = [0.2729])
                                    ]),mode='test',cv=k)

        test_loader = DataLoader(testset,shuffle=False,batch_size=1,num_workers=2)
        model = densenet121()
        densenet = densenet121_RPN()

        localization = DLA_model()
        densenet.load_state_dict(torch.load('/home/junkyu/project/tumor_detection/densenet_RPN/model/RPN3.pth'))

        densenet.eval()
        model.load_state_dict(torch.load('/home/junkyu/project/tumor_detection/mixup_model/model/DLA_segheat_cls_end2_'+str(k)+'.pth'))
        localization.load_state_dict(torch.load('/home/junkyu/project/tumor_detection/mixup_model/model/DLA_segheat_local_end2_'+str(k)+'.pth'))
        model = model.to(0)
        localization = DataParallel(localization).to(0)
        densenet = densenet.to(0)
        model.eval()
        localization.eval()
        ff,div = 0,0
        zz,zo,oz,oo = 0,0,0,0
        sig=50
        asdf = 0
        conf = torch.zeros((4,4))
        tsne_fig = torch.tensor([])
        label = torch.tensor([])
        mix_check = False
        with torch.no_grad():
            for i,(input,target,mask,name) in enumerate(test_loader):
                if name[0] in total:
                    input, target, mask = input.to(0), target.to(0), mask.to(0)
                    ######################################################### mix up
                    if mix_check == True:
                        input, hard_target, mask, mix, name = mixup(input,target,mask, input.shape[0],name)
                    else:
                        hard_target = target
                        mix = torch.zeros((input.shape[0])).to(0)
                    # input, targets_a, targets_b, lam = mixup_data(input, target,
                    #                                            1.0, True)
                    ##########################################################
                    # 8등분 split 해야됨 여기서..
                    patches = input.unfold(2, 375, 375).unfold(3,375,375)
                    patches = patches.contiguous().view(input.shape[0],3, -1, 375,375).transpose(1,2)
                    patches = patches.reshape(-1,3,375,375)
                    # patches의 차원은 48 8 3 375 375
                    ####################################################
                    input = F.interpolate(input, size = (375,750))
                    mask = F.interpolate(mask, size = (375,750))
                    #####################################################
                    x_cen, y_cen = torch.zeros(input.shape[0]).int(),torch.zeros(input.shape[0]).int()
                    predict_hard, predict_mix, predict_feature = model(patches, densenet)
                    predict_patch = localization(input, predict_feature)
                    pred = torch.argmax(predict_hard,1)
                    pred_mix = torch.argmax(predict_mix,1)

                    if pred_mix == mix:
                        asdf += 1
                    ##########################

                    conf[target,pred] += 1

                    if pred == target:
                        # vis.image(input[0,...]*0.2729 + 0.2602)
                        # vis.image(mask)
                        # vis.image((scale(predict_patch)>0.5).float())
                        # vis.image(scale(predict_patch))

                        temp = (scale(predict_patch)>0.5).float()
                        inter = ((mask + temp) == 2).sum()
                        uni = ((mask + temp) == 1).sum()
                        zz = inter/(inter+uni)
                        kk.append(zz.cpu())
                        ff += 1
                    div += 1
        print(conf)
        toto += conf
    print(toto)
    print(zz / ff)
    print('sadfsadfsdf')
    print(asdf / div)
    print('sadfjhjkhjk')
    print(ff / div)
    print(len(kk))
