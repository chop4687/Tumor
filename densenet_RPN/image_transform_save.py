import torch
import os
from PIL import Image, ImageChops
from torchvision import transforms as T
from utils.etc import find_box
import random
import sys
if __name__ == '__main__':
    #################################
    # # bounding box 값 txt 저장하기
    import cv2
    import numpy as np
    root = '/home/junkyu/project/tumor_detection/unet/newtumor_segmentation/'
    img_name = os.listdir(root)
    for i in img_name:
        img = cv2.imread(root+i,cv2.IMREAD_GRAYSCALE)
        ret, labels = cv2.connectedComponents(img)
        # z = torch.zeros((750,1500))

        # from visdom import Visdom
        # vis = Visdom(port=13246, env='test')
        # vis.image(T.ToTensor()(img))
        temp = np.zeros((ret-1,4))
        for h in range(1,ret):
            k = torch.where(T.ToTensor()(labels)==h)
            if k[1].max().item() - k[1].min().item() == 0:
                h -= 1
                temp = temp[h-1,:]
                temp = temp[np.newaxis,:]
                continue
            temp[h-1,0] = k[1].min().item()
            temp[h-1,1] = k[2].min().item()
            temp[h-1,2] = k[1].max().item()
            temp[h-1,3] = k[2].max().item()
        np.save('./data/bbox_idx2/'+i[:-4]+'.npy',temp)
    #     z[k[1].min():k[1].max(),k[2].min():k[2].max()] = 1
    #     sys.stdout = open('./bbox_idx/'+i[:-4]+'.txt','a')
    #     print(k[1].min().item(),k[2].min().item(),k[1].max().item(),k[2].max().item())
    #################################

    ##################################
    # 1500x750으로 bbox patch 따로 저장
    # root = '/home/junkyu/project/re_tumor/data/tumor_bbox/'
    # img_name = os.listdir(root)
    # for i in img_name:
    #     img = Image.open(root+i).convert('RGB')
    #     img = img.resize((1500,750))
    #     img.save('/home/junkyu/project/re_tumor/RPN/data/tumor_bbox/'+i[:-4]+'.png')
    ###################################

    ##################################
    #black img 저장
    root = '/home/junkyu/project/re_tumor/RPN/data/tumor_bbox/'
    img_name = os.listdir(root)
    for i in img_name:
        img = Image.open(root+i).convert('RGB')
        img = T.ToTensor()(img)
        temp = torch.where(img<1)
        x_min = torch.min(temp[1])
        x_max = torch.max(temp[1])
        y_min = torch.min(temp[2])
        y_max = torch.max(temp[2])
        black_img = torch.zeros((750,1500))
        black_img[x_min:x_max, y_min:y_max] = 1
        temp = T.ToPILImage()(black_img)
        temp.save('/home/junkyu/project/re_tumor/RPN/data/black/'+i[:-4]+'.png')
    ######################################

    #######################################
    # img * mask 저장
    # root = '/home/junkyu/project/re_tumor/data/dataset/tumor/'
    # root1 = '/home/junkyu/project/re_tumor/data/'
    # img_name = os.listdir(root)
    # for i in img_name:
    #     img = Image.open(os.path.join(root1, 'dataset/tumor', i)).convert('RGB').resize((1500,750))
    #     mask = Image.open(os.path.join(root1, 'mask/tumor', i[:-4]+'.png')).convert('RGB').resize((1500,750))
    #     mask_img = ImageChops.multiply(img, mask)
    #     mask_img.save('/home/junkyu/project/re_tumor/RPN/data/mask_img/'+i[:-4]+'.png')
    #####################################

    ######################################
    # img data tif to png
    # root = '/home/junkyu/project/re_tumor/RPN/data/dataset/ttt/'
    # img_name = os.listdir(root)
    # for i in img_name:
    #     img = Image.open(root+i).convert('RGB')
    #     img.save('/home/junkyu/project/re_tumor/RPN/data/dataset/tumor/'+i[:-4]+'.png')
    #######################################

    ########################################
    # patch 짤라서 저장하기
    # root = '/home/junkyu/project/re_tumor/RPN/data/'
    # img_name = os.listdir(root+'mask_img/')
    # kk = 0
    # for i in img_name:
    #     mask_img = Image.open(root+'mask_img/'+i)
    #     black_img = Image.open(root+'black/'+i)
    #     x_min, y_min, x_max, y_max = find_box(black_img)
    #     for l in range(10):
    #         x = random.randint(x_min, x_max)
    #         y = random.randint(y_min, y_max)
    #         x = max(0,min(x,750))
    #         y = max(0,min(y,1500))
    #         temp = mask_img.crop((y-187, x-187, y+188, x+188))
    #         temp_black = black_img.crop((y-187, x-187, y+188, x+188))
    #         temp.save(root+'patch/image/'+str(kk)+'.png')
    #         temp_black.save(root+'patch/label/'+str(kk)+'.png')
    #         kk += 1
    #
    #     x = random.randint(0, 750 - 375)
    #     y = random.randint(0, 1500 - 375)
    #     temp = mask_img.crop((y, x, y + 375, x + 375))
    #     temp_black = black_img.crop((y, x, y + 375, x + 375))
    #     temp.save(root+'patch/image/'+str(kk)+'.png')
    #     temp_black.save(root+'patch/label/'+str(kk)+'.png')
    #     kk += 1
    ########################################

    ###########################################
    # IoU test
    # from utils.etc import IoU, IoU2
    # anchor = torch.randn(2,50,4)
    # gt = torch.randn(2,4)
    # m_gt = gt.expand(50,2,4).transpose(0,1)
    # for i in range(2):
    #     print(IoU(anchor[i,...],m_gt[i,...]))
    # print("-------------")
    # for i in range(2):
    #     for j in range(50):
    #         print(IoU2(anchor[i,j,...],m_gt[i,j,...]))
    # print("-------------fdgsfdgsfdgsfdg")
    # value = torch.zeros((2,50))
    # keep = torch.zeros((2,50))
    # bat = 0
    # for bat in range(2):
    #     print(bat)
    #     temp = IoU(anchor[bat,...],m_gt[bat,...])
    #     value[bat,...] = temp
    #     #keep = value.clone()
    #     keep[bat, temp>0.05] = 1
    #     keep[bat, temp<=0.05] = 0
    # print(keep)
    # print("---asdfsadfsdafsdf----------")
    # keep = torch.zeros((2,50))
    # value = torch.zeros((2,50))
    # for i in range(2):
    #     for bbox in range(50):
    #         temp = IoU2(anchor[i,bbox,:],gt[i,:])
    #         value[i,bbox] = temp
    #         if temp >= 0.05:
    #             keep[i,bbox] = 1
    #         else:
    #             keep[i,bbox] = 0
    # print(keep)
    #################################
