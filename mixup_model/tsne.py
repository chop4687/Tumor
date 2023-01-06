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
    tsne_fig = torch.tensor([])
    label = torch.tensor([])
    for k in range(1,2):
        os.environ['CUDA_VISIBLE_DEVICES'] = '6,2'
        vis = Visdom(port = 13246,env='test3')
        testset = tumor(root = '/home/junkyu/project/tumor_detection/DLA_model/data200',transform = T.Compose([
                                    ToTensor(),
                                    Normalize(mean = [0.2602], std = [0.2729])
                                    ]),mode='test',cv=k)

        test_loader = DataLoader(testset,shuffle=False,batch_size=10,num_workers=2)
        model = densenet121()
        densenet = densenet121_RPN()

        localization = DLA_model()
        densenet.load_state_dict(torch.load('/home/junkyu/project/tumor_detection/densenet_RPN/model/RPN3.pth'))

        densenet.eval()
        model.load_state_dict(torch.load('/home/junkyu/project/tumor_detection/mixup_model/model/DLA_segheat_cls_hard3_'+str(k)+'.pth'))
        localization.load_state_dict(torch.load('/home/junkyu/project/tumor_detection/mixup_model/model/DLA_segheat_local_hard3_'+str(k)+'.pth'))
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
        tsne = TSNE(n_components=2,perplexity=30)
        mix_check = False
        with torch.no_grad():
            for i,(input,target,mask,name) in enumerate(test_loader):
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
                predict_hard, predict_mix, predict_feature,img_class = model(patches, densenet)
                tsne_fig = torch.cat((tsne_fig,img_class.cpu()))
                label = torch.cat((label,target.cpu()))

    tsne_ref = tsne.fit_transform(tsne_fig.cpu())
    plt.scatter(tsne_ref[:, 0], tsne_ref[:, 1], c=label)
    plt.savefig('output_2d.png', bbox_inches='tight')
