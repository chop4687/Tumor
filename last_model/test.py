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
import torchvision
def scale(ten):
    return (ten - ten.min()) / (ten.max() - ten.min())

if __name__ == '__main__':
    toto = torch.zeros((4,4))
    pp, pre, tar = [], [], []
    for k in range(5):
        os.environ['CUDA_VISIBLE_DEVICES'] = '2'
        vis = Visdom(port = 13246,env='test')
        testset = tumor(root = '/home/junkyu/project/tumor_detection/last_model/data200',transform = T.Compose([
                                    ToTensor(),
                                    Normalize(mean = [0.2602,0.2602,0.2602], std = [0.2729,0.2729,0.2729])
                                    ]),mode='test',cv=k)

        test_loader = DataLoader(testset,shuffle=False,batch_size=1,num_workers=2,collate_fn=make_batch)
        model = densenet121()
        densenet = densenet121_RPN()
        # temp_model = torchvision.models.densenet121()
        # temp_model.classifier = nn.Linear(1024,3)
        # temp_model.load_state_dict(torch.load('/home/junkyu/project/tumor_detection/default_model/three.pth'))
        # densenet.first_conv.conv0 = temp_model.features.conv0
        # densenet.first_conv.norm0 = temp_model.features.norm0
        # densenet.first_conv.relu0 = temp_model.features.relu0
        # densenet.first_conv.pool0 = temp_model.features.pool0
        #
        # densenet.block1[0] = temp_model.features.denseblock1
        # densenet.block1[1] = temp_model.features.transition1
        #
        # densenet.block2[0] = temp_model.features.denseblock2
        # densenet.block2[1] = temp_model.features.transition2
        #
        # densenet.block3[0] = temp_model.features.denseblock3
        # densenet.block3[1] = temp_model.features.transition3
        #
        # densenet.block4[0] = temp_model.features.denseblock4
        densenet.load_state_dict(torch.load('/home/junkyu/project/tumor_detection/densenet_RPN/model/RPN3.pth'))
        # for p in densenet.parameters():
        #     p.requires_grad = False
        densenet.eval()
        model_unet = UNet(n_channels = 256, n_classes = 1)
        model.load_state_dict(torch.load('/home/junkyu/project/tumor_detection/last_model/model/four_final_'+str(k)+'.pth'))
        model_unet.load_state_dict(torch.load('/home/junkyu/project/tumor_detection/last_model/model/four_final'+str(k)+'.pth'))
        model = DataParallel(model).to(0)
        model_unet = DataParallel(model_unet).to(0)
        densenet = DataParallel(densenet).to(0)
        model.eval()
        model_unet.eval()
        ff,div = 0,0
        zz,zo,oz,oo = 0,0,0,0
        conf = torch.zeros((4,4))
        with torch.no_grad():
            for i,(input,class_y,mask, name, pixel) in enumerate(test_loader):
                input, class_y, mask = input.to(0), class_y.to(0), mask.to(0)
                predict_img, predict_patch = model(input,densenet)
                predict_patch = model_unet(predict_patch)
                name = name[::8]
                pixel = pixel[::8]
                class_y = class_y[::8]
                mask = mask.unsqueeze(1)
                mask = torch.nn.functional.interpolate(mask, size = (11,22))
                mask = mask[::8]
                pred = torch.argmax(predict_img,1)
                vis.image(nn.functional.interpolate(predict_patch, size = (448,896)))
                ##########################
                if class_y != 0:
                    pp.append(pixel)
                    pre.append(pred)
                    tar.append(class_y)
                ##########################
                conf[class_y,pred] += 1
                #print(pred.item(),class_y.item())
                if class_y == pred:
                    #if pred == 0:
                    ff += 1
                div += 1
        print(conf)
        toto += conf

    vals = np.array(pp).reshape(-1)
    sort_index = np.argsort(vals)
    #print(sort_index)
    pre = np.array(pre)
    tar = np.array(tar)
    # pre = pre[sort_index]
    # tar = tar[sort_index]
    y = []
    for i in range(len(vals)):
        print(pre[i],tar[i])
        if pre[i] == tar[i]:
            y.append(1)
        else:
            y.append(0)
        #vis.line(X=[i], Y=[y], win='size', name='size', update='append', opts=dict(showlegend=True, title='size'))
    y = np.array(y)[sort_index]
    for i in range(13,len(vals)):
        vis.scatter(X=[i-13], Y=[y[i]], win='size', name='acc', update='append', opts=dict(showlegend=True, title='size'))
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
