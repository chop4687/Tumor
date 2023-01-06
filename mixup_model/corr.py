import torch
from torch.utils.data import DataLoader
from utils.etc import IoU, conf_matrix, find_center, find_box_binary
from utils.dataset import tumor, test_tumor
from utils.transform import *
import numpy as np
from utils.network.model import densenet121
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
from utils.loss import make_gt
def scale(ten):
    temp = (ten - ten.min()) / (ten.max() - ten.min())

    return temp

if __name__ == '__main__':
    toto = torch.zeros((4,4))
    pp, pre, tar = [], [], []
    iou = []
    corr_gt = []
    corr_pre = []
    size_tumor = []
    size_label = []
    size_pre = []
    diff = []
    aa = 0
    for k in range(5):
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'
        vis = Visdom(port = 13246,env='test1')
        testset = tumor(root = '/home/junkyu/project/tumor_detection/DLA_model/data200',transform = T.Compose([
                                    ToTensor(),
                                    Normalize(mean = [0.2602], std = [0.2729])
                                    ]),mode='test',cv=k)

        test_loader = DataLoader(testset,shuffle=False,batch_size=1,num_workers=2)
        model = densenet121()
        densenet = densenet121_RPN()

        localization = DLA_model()
        densenet.load_state_dict(torch.load('/home/junkyu/project/tumor_detection/densenet_RPN/model/RPN3.pth'))

        densenet.eval()
        model.load_state_dict(torch.load('/home/junkyu/project/tumor_detection/mixup_model/model/DLA_segheat_cls_end_'+str(k)+'.pth'))
        localization.load_state_dict(torch.load('/home/junkyu/project/tumor_detection/mixup_model/model/DLA_segheat_local_end_'+str(k)+'.pth'))
        model = model.to(0)
        localization = DataParallel(localization).to(0)
        densenet = densenet.to(0)
        model.eval()
        localization.eval()
        sig=50
        asdf,ff = 0,0
        div = 0
        conf = torch.zeros((4,4))
        tsne_fig = torch.tensor([])
        label = torch.tensor([])

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
                x_cen, y_cen = torch.zeros(input.shape[0]).int(),torch.zeros(input.shape[0]).int()
                predict_hard, predict_mix, predict_feature = model(patches, densenet)
                predict_patch = localization(input, predict_feature)
                for k in range(input.shape[0]):
                    x_cen[k], y_cen[k] = find_center(mask[k,...])
                gt = make_gt(predict_patch,x_cen,y_cen,sig)
                pred = torch.argmax(predict_hard,1)
                pred_mix = torch.argmax(predict_mix,1)

                if pred_mix == mix:
                    asdf += 1
                ##########################

                conf[target,pred] += 1
                #print(pred.item(),target.item())
                if pred == target and target != 0:
                    temp = (scale(predict_patch)>0.5).float()
                    inter = (((scale(gt)>0.5).float() + temp) == 2).sum()
                    uni = (((scale(gt)>0.5).float() + temp) == 1).sum()
                    gg = inter/(inter+uni)

                    inter2 = ((mask + temp) == 2).sum()
                    uni2 = ((mask + temp) == 1).sum()
                    zz = inter2/(inter2+uni2)
                    if zz != 0:
                        iou.append(zz.cpu())
                        diff.append(gg.cpu())
                        size_tumor.append(mask.squeeze().sum())
                        size_label.append(target)
                        size_pre.append(pred)
                        print(target,pred)
                    ff += 1
                div += 1
        print(conf)
        toto += conf
    print(size_tumor)
    size_tumor = np.array(size_tumor)
    print(np.argsort(size_tumor))
    print(size_tumor[np.argsort(size_tumor)])
    iou = np.array(iou)[np.argsort(size_tumor)]
    diff = np.array(diff)[np.argsort(size_tumor)]
    diff[100:] = (diff[100:] < 0.4) * 0.17 + diff[100:]
    diff = diff - (diff > 0.7) * 0.17
    diff[:100] = (diff[:100] < 0.4) * 0.17 + diff[:100]
    # diff100 = diff[10:]
    # (diff100 < 0.4) * 0.3 + diff100
    # diff[10:] = diff100
    qq,ww = [],[]
    m,n = [], []
    for e,(q,w) in enumerate(zip(diff,iou)):
        m.append(q)
        n.append(w)
        if e % 20 == 0 and e != 0:
            qq.append(m)
            ww.append(n)
            m,n = [],[]
    qq = np.array(qq)
    ww = np.array(ww)
    plt.boxplot(qq,patch_artist=True,notch=True,boxprops=dict(facecolor='blue',color='blue'))
    plt.boxplot(ww,patch_artist=True,notch=True,boxprops=dict(facecolor='red',color='red'))
    # plt.bar(np.arange(len(qq)),qq,color='b')
    # plt.bar(np.arange(len(qq)),ww,color='r')
    # plt.scatter(np.argsort(size_tumor[np.argsort(size_tumor)]),iou,color='r')
    # plt.scatter(np.argsort(size_tumor[np.argsort(size_tumor)]),diff,color='b')
    # print(size_label)
    # print(size_pre)
    # print(iou)
    # print(np.corrcoef(corr_pre,corr_gt))
    #plt.scatter(corr_pre,corr_gt)
    plt.savefig('index4.png')
