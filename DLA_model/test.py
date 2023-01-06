import torch
from torch.utils.data import DataLoader
from utils.etc import make_batch, IoU, conf_matrix
from utils.dataset import tumor
from utils.transform import *
import numpy as np
from utils.network.model import densenet121
from visdom import Visdom
import os
from torch.nn.parallel.data_parallel import DataParallel
from utils.network.unet import UNet
from utils.network.rpn import densenet121_RPN
from utils.loss import MultiTaskLoss
import torchvision
import random
def scale(ten):
    temp = (ten - ten.min()) / (ten.max() - ten.min())
    result = torch.zeros_like(temp)
    result[temp==1] += 2
    result = result.squeeze()
    idx = [result.max(dim=0)[1].max(),result.max(dim=1)[1].max()]
    i_idx = random.randint(4,7)
    j_idx = random.randint(4,7)
    for i in range(i_idx):
        for j in range(j_idx):
            rand = random.uniform(0,1)
            result[(idx[0]),(idx[1])] += rand
        #result[(idx[0]-3):(idx[0]+3),(idx[1]-3):(idx[1]+3)] += rand
    result = (result - result.min()) / (result.max() - result.min())
    result[result < 0.3] = 0
    result = result.unsqueeze(0).unsqueeze(0)
    return result

if __name__ == '__main__':
    toto = torch.zeros((4,4))
    pp, pre, tar = [], [], []
    for k in range(0,5):
        os.environ['CUDA_VISIBLE_DEVICES'] = '8'
        vis = Visdom(port = 13246,env='test')
        testset = tumor(root = '/home/junkyu/project/tumor_detection/DLA_model/data200',transform = T.Compose([
                                    ToTensor(),
                                    Normalize(mean = [0.2602,0.2602,0.2602], std = [0.2729,0.2729,0.2729])
                                    ]),mode='test',cv=k)

        test_loader = DataLoader(testset,shuffle=False,batch_size=1,num_workers=2,collate_fn=make_batch)
        model = densenet121()
        densenet = densenet121_RPN()

        densenet.load_state_dict(torch.load('/home/junkyu/project/tumor_detection/densenet_RPN/model/RPN3.pth'))
        # for p in densenet.parameters():
        #     p.requires_grad = False
        densenet.eval()
        model_unet = UNet(n_channels = 256, n_classes = 1)
        model.load_state_dict(torch.load('/home/junkyu/project/tumor_detection/DLA_model/model/01_final_'+str(k)+'.pth'))
        #model_unet.load_state_dict(torch.load('/home/junkyu/project/tumor_detection/DLA_model/model/final'+str(k)+'.pth'))
        model = DataParallel(model).to(0)
        #model_unet = DataParallel(model_unet).to(0)
        densenet = DataParallel(densenet).to(0)
        model.eval()
        model_unet.eval()
        ff,div = 0,0
        zz,zo,oz,oo = 0,0,0,0
        conf = torch.zeros((4,4))
        with torch.no_grad():
            for i,(big_img, input,class_y,mask,name) in enumerate(test_loader):
                input, class_y, mask = input.to(0), class_y.to(0), mask.to(0)
                predict_img, predict_patch = model(big_img, input, densenet)
                class_y = class_y[::8]
                name = name[::8]

                mask = mask.unsqueeze(1)
                mask = nn.functional.interpolate(mask, size = (22,44))
                mask = mask[::8]
                pred = torch.argmax(predict_img,1)
                ##########################

                conf[class_y,pred] += 1
                #print(pred.item(),class_y.item())
                if class_y == pred and class_y != 0:
                    vis.image(nn.functional.interpolate(mask, size = (448,896)))
                    temp = nn.functional.interpolate(scale(predict_patch), size = (448,896))
                    #temp = torchvision.transforms.GaussianBlur(3)(temp)
                    vis.image(temp)
                    # print(mask.shape)
                    # print(input.shape)
                    # print(predict_patch.shape)
                    # print(torch.unique(mask))
                    # vis.image(nn.functional.interpolate(mask, size = (448,896)))
                    # #for j in range(8):
                    #     #vis.image(input[j,...]*0.2729 + 0.2602)
                    #
                    # vis.image(nn.functional.interpolate(predict_patch, size = (448,896)))
                    # exit()
                    #if pred == 0:
                    ff += 1
                div += 1
        print(conf)
        toto += conf
    print(toto)
                ###################################
            # if class_y == 1:
            #     #if IoU(predict_patch[0],mask[0]) > 0:
            #     temp = Image.open('/home/junkyu/project/re_tumor/data/dataset/tumor/' + name[0]).resize((400,200))
            #     vis.image(T.ToTensor()(temp))
            #     vis.image(torch.nn.functional.interpolate(scale(predict_patch[0]).unsqueeze(0),size = (200, 400)))
            #     vis.image(torch.nn.functional.interpolate(mask[0].unsqueeze(0),size = (200, 400)))
            #     # print(torch.nn.Softmax(dim=1)(predict_img))
            #     # print(IoU(predict_patch[0],mask[0]))
            #     # print(class_y)
            #     #ff += IoU(predict_patch[0],mask[0])
            #     div += 1
            ####################################
                # exit()
            # if class_y == 0 and pred == 0:
            #     zz += 1
            # elif class_y == 0 and pred == 1:
            #     zo += 1
            #     temp = Image.open('/home/junkyu/project/re_tumor/data/dataset/normal/' + name[0]).resize((400,200))
            #     vis.image(T.ToTensor()(temp))
            #     vis.image(torch.nn.functional.interpolate(predict_patch[0].unsqueeze(0),size = (200, 400)))
            #     vis.image(torch.nn.functional.interpolate(mask[0].unsqueeze(0),size = (200, 400)))
            #     ff += IoU(predict_patch[0],mask[0])
            #     div += 1
            #     print(torch.nn.Softmax(dim=1)(predict_img))
            #     print(IoU(predict_patch[0],mask[0]))
            #     print(class_y)
            # elif class_y == 1 and pred == 0:
            #     oz += 1
            #     temp = Image.open('/home/junkyu/project/re_tumor/data/dataset/tumor/' + name[0]).resize((400,200))
            #     vis.image(T.ToTensor()(temp))
            #     vis.image(torch.nn.functional.interpolate(predict_patch[0].unsqueeze(0),size = (200, 400)))
            #     vis.image(torch.nn.functional.interpolate(mask[0].unsqueeze(0),size = (200, 400)))
            #     ff += IoU(predict_patch[0],mask[0])
            #     div += 1
            #     print(torch.nn.Softmax(dim=1)(predict_img))
            #     print(IoU(predict_patch[0],mask[0]))
            #     print(class_y)
            # else:
            #     oo += 1
            # if class_y == 1:
            #     temp = Image.open('/home/junkyu/project/re_tumor/data/dataset/tumor/' + name[0]).resize((400,200))
            #     vis.image(T.ToTensor()(temp))
            #     vis.image(torch.nn.functional.interpolate(predict_patch[0].unsqueeze(0),size = (200, 400)))
            #     vis.image(torch.nn.functional.interpolate(mask[0].unsqueeze(0),size = (200, 400)))
            #     ff += IoU(predict_patch[0],mask[0])
            #     div += 1
            #     print(torch.nn.Softmax(dim=1)(predict_img))
            #     print(IoU(predict_patch[0],mask[0]))
            #     print(class_y)
            # else:
            #     temp = Image.open('/home/junkyu/project/re_tumor/data/dataset/normal/' + name[0]).resize((400,200))

    #         #break
    print(ff / div)
    # print(f'{zz},{zo}\n{oz},{oo}')
