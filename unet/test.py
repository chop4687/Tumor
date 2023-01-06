import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from utils.network.unet_encoder import Unet_encoder
from utils.network.unet_decoder import Unet_decoder
#from transform import Resize
from utils.dataset import pretrain_sample
from torchvision import transforms as T
from visdom import Visdom
from torch.nn.parallel.data_parallel import DataParallel
import torch.nn.functional as F

from PIL import Image,ImageChops

import cv2

def change_size(img, size=(512,1024)):
    out = F.interpolate(img,
                        size,
                        mode='bilinear',
                        align_corners=True)
    return out

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4'
    vis = Visdom(port=13246,env='connect')

    test_set = pretrain_sample(root = '/home/junkyu/data/1700_tumor_annotation/add_DC/',transform=T.Compose([T.Resize((512,1024)),T.ToTensor()]),train='test')
                                                                                    #T.Lambda(lambda x: (T.ToTensor()(x[0]), T.ToTensor()(x[1])))]), train = None)
    test_loader = DataLoader(test_set,batch_size = 1,shuffle = False)

    encoder_net1 = Unet_encoder(n_channels=1)
    decoder_net1 = Unet_decoder(n_classes=1)
    encoder_net1.load_state_dict(torch.load("/home/junkyu/project/tumor_detection/unet/model/Unet_encoder_default_1"))
    decoder_net1.load_state_dict(torch.load("/home/junkyu/project/tumor_detection/unet/model/Unet_decoder_default_1"))

    encoder_net2 = Unet_encoder(n_channels=1)
    decoder_net2 = Unet_decoder(n_classes=1)
    encoder_net2.load_state_dict(torch.load("/home/junkyu/project/tumor_detection/unet/model/Unet_encoder_default_2"))
    decoder_net2.load_state_dict(torch.load("/home/junkyu/project/tumor_detection/unet/model/Unet_decoder_default_2"))

    encoder_net3 = Unet_encoder(n_channels=1)
    decoder_net3 = Unet_decoder(n_classes=1)
    encoder_net3.load_state_dict(torch.load("/home/junkyu/project/tumor_detection/unet/model/Unet_encoder_default_3"))
    decoder_net3.load_state_dict(torch.load("/home/junkyu/project/tumor_detection/unet/model/Unet_decoder_default_3"))

    encoder_net4 = Unet_encoder(n_channels=1)
    decoder_net4 = Unet_decoder(n_classes=1)
    encoder_net4.load_state_dict(torch.load("/home/junkyu/project/tumor_detection/unet/model/Unet_encoder_default_4"))
    decoder_net4.load_state_dict(torch.load("/home/junkyu/project/tumor_detection/unet/model/Unet_decoder_default_4"))

    encoder_net5 = Unet_encoder(n_channels=1)
    decoder_net5 = Unet_decoder(n_classes=1)
    encoder_net5.load_state_dict(torch.load("/home/junkyu/project/tumor_detection/unet/model/Unet_encoder_default_5"))
    decoder_net5.load_state_dict(torch.load("/home/junkyu/project/tumor_detection/unet/model/Unet_decoder_default_5"))
    all = 0.0
    total1, correct1, total2, correct2 = 0.0,0.0,0.0,0.0
    for i,(input,name) in enumerate(test_loader):
        x1,x2,x3,x4,x5 = encoder_net1(input)
        predict1,predict11,predict111 = decoder_net1(x1=x1,x2=x2,x3=x3,x4=x4,x5=x5)

        x1,x2,x3,x4,x5 = encoder_net2(input)
        predict2,predict22,predict222 = decoder_net2(x1=x1,x2=x2,x3=x3,x4=x4,x5=x5)

        x1,x2,x3,x4,x5 = encoder_net3(input)
        predict3,predict33,predict333 = decoder_net3(x1=x1,x2=x2,x3=x3,x4=x4,x5=x5)

        x1,x2,x3,x4,x5 = encoder_net4(input)
        predict4,predict44,predict444 = decoder_net4(x1=x1,x2=x2,x3=x3,x4=x4,x5=x5)

        x1,x2,x3,x4,x5 = encoder_net5(input)
        predict5,predict55,predict555 = decoder_net5(x1=x1,x2=x2,x3=x3,x4=x4,x5=x5)

        predict1, predict11 = change_size(predict1), change_size(predict11)
        predict2, predict22 = change_size(predict2), change_size(predict22)
        predict3, predict33 = change_size(predict3), change_size(predict33)
        predict4, predict44 = change_size(predict4), change_size(predict44)
        predict5, predict55 = change_size(predict5), change_size(predict55)

        temp1 = predict1 + predict11 + predict111
        temp1 = torch.clamp(temp1,max=1)
        temp1[0,...][temp1[0,...] >= 0.5] = 1
        temp1[0,...][temp1[0,...] != 1] = 0
        #tensor 1,1,512,1024

        temp1 = (temp1[0,...]*255).type(torch.uint8).squeeze(0).detach().cpu().numpy()

        label,img1 = cv2.connectedComponents(temp1)
        from torchvision.utils import save_image
        img1 = T.ToTensor()(img1).type(torch.float32)
        print(name[0])
        save_image(img1,'./newdata/mask/'+name[0])
        ttt = input[0,0,...] * img1
        ttt = T.ToPILImage()(ttt)
        img1 = T.ToPILImage()(img1)
        input = T.ToPILImage()(input[0,...])

        ttt.save('./newdata/mask_img/'+name[0])

        input.save('./newdata/img/'+name[0])


###################################
# import os
# import sys
# import numpy as np
# import torch
# from torch.utils.data import Dataset, DataLoader, ConcatDataset
#
# from unet_divide import Unet_encoder
# from unet_divide import Unet_decoder
# from transform import Resize
# from data import pretrain_sample
# from torchvision import transforms as T
# from visdom import Visdom
# from torch.nn.parallel.data_parallel import DataParallel
# import torch.nn.functional as F
#
# from PIL import Image,ImageChops
#
# import cv2
#
# def change_size(img, size=(512,1024)):
#     out = F.interpolate(img,
#                         size,
#                         mode='bilinear',
#                         align_corners=True)
#     return out
#
# if __name__ == '__main__':
#     os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4'
#     vis = Visdom(port=13246,env='connect')
#
#     test_set = pretrain_sample(root = '/home/junkyu/data/1700_tumor_annotation',transform=T.Compose([Resize((512,1024)),
#                                                                                     T.Lambda(lambda x: (T.ToTensor()(x[0]), T.ToTensor()(x[1])))]), train = None)
#
#     test_loader = DataLoader(test_set,batch_size = 1,shuffle = False)
#
#     encoder_net1 = Unet_encoder(n_channels=1)
#     decoder_net1 = Unet_decoder(n_classes=1)
#     encoder_net1.load_state_dict(torch.load("/home/junkyu/project/tumor_detection/model/Unet_encoder_default_1"))
#     decoder_net1.load_state_dict(torch.load("/home/junkyu/project/tumor_detection/model/Unet_decoder_default_1"))
#
#     encoder_net2 = Unet_encoder(n_channels=1)
#     decoder_net2 = Unet_decoder(n_classes=1)
#     encoder_net2.load_state_dict(torch.load("/home/junkyu/project/tumor_detection/model/Unet_encoder_default_2"))
#     decoder_net2.load_state_dict(torch.load("/home/junkyu/project/tumor_detection/model/Unet_decoder_default_2"))
#
#     encoder_net3 = Unet_encoder(n_channels=1)
#     decoder_net3 = Unet_decoder(n_classes=1)
#     encoder_net3.load_state_dict(torch.load("/home/junkyu/project/tumor_detection/model/Unet_encoder_default_3"))
#     decoder_net3.load_state_dict(torch.load("/home/junkyu/project/tumor_detection/model/Unet_decoder_default_3"))
#
#     encoder_net4 = Unet_encoder(n_channels=1)
#     decoder_net4 = Unet_decoder(n_classes=1)
#     encoder_net4.load_state_dict(torch.load("/home/junkyu/project/tumor_detection/model/Unet_encoder_default_4"))
#     decoder_net4.load_state_dict(torch.load("/home/junkyu/project/tumor_detection/model/Unet_decoder_default_4"))
#
#     encoder_net5 = Unet_encoder(n_channels=1)
#     decoder_net5 = Unet_decoder(n_classes=1)
#     encoder_net5.load_state_dict(torch.load("/home/junkyu/project/tumor_detection/model/Unet_encoder_default_5"))
#     decoder_net5.load_state_dict(torch.load("/home/junkyu/project/tumor_detection/model/Unet_decoder_default_5"))
#     all = 0.0
#     total1, correct1, total2, correct2 = 0.0,0.0,0.0,0.0
#     for i,(input, label) in enumerate(test_loader):
#         x1,x2,x3,x4,x5 = encoder_net1(input)
#         predict1,predict11,predict111 = decoder_net1(x1=x1,x2=x2,x3=x3,x4=x4,x5=x5)
#
#         x1,x2,x3,x4,x5 = encoder_net2(input)
#         predict2,predict22,predict222 = decoder_net2(x1=x1,x2=x2,x3=x3,x4=x4,x5=x5)
#
#         x1,x2,x3,x4,x5 = encoder_net3(input)
#         predict3,predict33,predict333 = decoder_net3(x1=x1,x2=x2,x3=x3,x4=x4,x5=x5)
#
#         x1,x2,x3,x4,x5 = encoder_net4(input)
#         predict4,predict44,predict444 = decoder_net4(x1=x1,x2=x2,x3=x3,x4=x4,x5=x5)
#
#         x1,x2,x3,x4,x5 = encoder_net5(input)
#         predict5,predict55,predict555 = decoder_net5(x1=x1,x2=x2,x3=x3,x4=x4,x5=x5)
#
#         predict1, predict11 = change_size(predict1), change_size(predict11)
#         predict2, predict22 = change_size(predict2), change_size(predict22)
#         predict3, predict33 = change_size(predict3), change_size(predict33)
#         predict4, predict44 = change_size(predict4), change_size(predict44)
#         predict5, predict55 = change_size(predict5), change_size(predict55)
#
#         temp1 = predict1 + predict11 + predict111
#         temp1 = torch.clamp(temp1,max=1)
#         temp1[0,...][temp1[0,...] >= 0.5] = 1
#         temp1[0,...][temp1[0,...] != 1] = 0
#         #tensor 1,1,512,1024
#
#         temp1 = (temp1[0,...]*255).type(torch.uint8).squeeze(0).detach().cpu().numpy()
#
#         label,img1 = cv2.connectedComponents(temp1)
#
#         img1 = T.ToTensor()(img1)
#         ttt = input[0,0,...] * img1
#         ttt = T.ToPILImage()(ttt)
#         # img1 = T.ToPILImage()(img1)
#         # input = T.ToPILImage()(input[0,...])
#
#         ttt.save('ff.png')
#
#         exit(0)
# ########################################
