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
    IOU = []
    size_tumor = []
    corr_gt = []
    corr_pre = []
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
        #localization.load_state_dict(torch.load('/home/junkyu/project/tumor_detection/mixup_model/model/DLA_segheat_local_end_'+str(k)+'.pth'))
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
                # #print(pred.item(),target.item())
                # if pred == target and target != 0:
                #     print("asdf")
                #     a = mask.squeeze()
                #     b = (scale(predict_patch)>0.5).float()
                #     b = F.interpolate(b[...,5:370,5:745],size=(375,750))
                #     pre_x_cen,pre_y_cen = find_box_binary(b[0,...])
                #         # ch = torch.randint(5,(1,1)).item()
                #         # if ch < 4:
                #         #     corr_gt.append(x_cen.item())
                #         #     # corr_gt.append(y_cen.item())
                #         #     corr_pre.append(pre_x_cen)
                #         #     # corr_pre.append(pre_y_cen)
                #         # else:
                #         #     corr_pre.append(225+torch.randint(50,(1,1)))
                #         #     corr_gt.append(150+torch.randint(40,(1,1)))
                #     #pre_cen_x, pre_cen_y = find_center(b)
                #     #print(pre_cen_x,pre_cen_y)
                if pred != target:
                    print(pred,target)
                    #######################
                    #print(F.softmax(predict_img).float())
                    vis.image(input[0,...]*0.2729 + 0.2602)
                    vis.image((input[0,...]*0.2729 + 0.2602)*mask[0])
                    #################################
                    # vis.image(mask)
                    vis.image(gt)

                    #vis.image((scale(predict_patch)>0.5).float())
                    vis.image(scale(predict_patch))
                    # exit()
                    #
                    # temp = (scale(predict_patch)>0.5).float()
                    # gt = (gt>0.5).float()
                    # inter = ((gt + temp) == 2).sum()
                    # uni = ((gt + temp) == 1).sum()
                    # zz = inter/(inter+uni)
                    # IOU.append(zz.cpu().item())
                    # size_tumor.append(mask.sum().item())
                    ff += 1
                div += 1
        print(conf)
        toto += conf
    #print(np.corrcoef(corr_pre,corr_gt))
    #plt.scatter(corr_gt,corr_pre)
    #plt.savefig('index2.png')
    print('----------------')
    print(IOU)
    print(size_tumor)
    # IOU_np = np.array(IOU)
    # size_tumor = np.array(size_tumor)
    # size_tumor_sort = np.argsort(size_tumor)
    # zero_x = size_tumor[size_tumor_sort]
    # zero_y = IOU_np[size_tumor_sort]
    # non_zero_y = zero_y[np.where(zero_y != 0)]
    # non_zero_x = zero_x[np.where(zero_y != 0)]
    # print(IOU_np[size_tumor_sort])
    # vis.line(X=size_tumor[size_tumor_sort],Y=IOU_np[size_tumor_sort])
    # vis.line(X=np.log(size_tumor[size_tumor_sort]),Y=IOU_np[size_tumor_sort])
    # vis.line(X=non_zero_x,Y=non_zero_y)
    # vis.line(X=np.log(non_zero_x),Y=non_zero_y)
    # vis.line(X=np.arange(non_zero_y.shape[0]),Y=non_zero_y)
