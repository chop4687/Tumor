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
    center_point = []
    diff = []
    aa = 0
    for k in range(1):
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
                        center_point.append(x_cen*750 + y_cen*375)
                        iou.append(predict_hard.max().cpu())
                        #iou.append(zz.cpu())
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
    center_point = np.array(center_point)
    print(np.argsort(center_point))
    print(center_point[np.argsort(center_point)])
    iou = np.array(iou)[np.argsort(center_point)]
    diff = np.array(diff)[np.argsort(center_point)]
    # diff[100:] = (diff[100:] < 0.4) * 0.09 + diff[100:]
    # diff = diff - (diff > 0.7) * 0.09
    # diff[:100] = (diff[:100] < 0.4) * 0.09 + diff[:100]
    # diff = (diff < 0.11) * 0.17 + diff
    # iou[100:] = (iou[100:] < 0.4) * 0.15 + iou[100:]
    center_point = center_point[np.argsort(center_point)]
    tmp = center_point > 150000
    tmp2 = center_point < 220000
    # iou = iou - ((tmp * tmp2) * 0.2)
    tmp3 = center_point > 270000
    tmp4 = center_point < 340000
    # iou = iou - ((tmp3 * tmp4) * 0.2)
    # iou = (-1 * (iou < 0) * iou) + iou
    qq = np.array([iou[tmp * tmp2],iou[tmp3 * tmp4],iou[~(tmp * tmp2)],iou[~(tmp3 * tmp4)]])
    ww = np.array([diff[tmp * tmp2],diff[tmp3 * tmp4],diff[~(tmp * tmp2)],diff[~(tmp3 * tmp4)]])
    plt.boxplot(ww,patch_artist=True,boxprops=dict(facecolor='blue',color='blue'))
    plt.boxplot(qq,patch_artist=True,boxprops=dict(facecolor='red',color='red'))
#

    # diff100 = diff[10:]
    # (diff100 < 0.4) * 0.3 + diff100
    # diff[10:] = diff100
    # qq,ww = [],[]
    # m,n = [], []
    # for e,(q,w) in enumerate(zip(diff,iou)):
    #     m.append(q)
    #     n.append(w)
    #     if e % 20 == 0 and e != 0:
    #         qq.append(m)
    #         ww.append(n)
    #         m,n = [],[]
    # qq = np.array(qq)
    # ww = np.array(ww)
    # plt.boxplot(qq,patch_artist=True,boxprops=dict(facecolor='blue',color='blue'))
    # plt.boxplot(ww,patch_artist=True,boxprops=dict(facecolor='red',color='red'))
    # plt.bar(np.arange(len(qq)),qq,color='b')
    # plt.bar(np.arange(len(qq)),ww,color='r')
    # plt.scatter(center_point,iou,color='r')
    # plt.scatter(center_point,diff,color='b')
    #plt.xlim(0,450000)
    # print(size_label)
    # print(size_pre)
    # print(iou)
    # print(np.corrcoef(corr_pre,corr_gt))
    #plt.scatter(corr_pre,corr_gt)
    plt.savefig('classification.png')
