import cv2
import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.utils import save_image
if __name__ == '__main__':
    ######################################################################
    # dir = os.listdir('/home/junkyu/project/tumor_detection/unet/newdata/img/')
    # for i in dir:
    #     print(i)
    #     img = cv2.imread('/home/junkyu/project/tumor_detection/unet/newdata/img/'+i)
    #     img = cv2.resize(img, dsize=(3000,1500))
    #     bbox = np.load('/home/junkyu/project/tumor_detection/densenet_RPN/data/bbox_idx/'+i[:-4]+'.npy')
    #     print(img.shape)
    #     print(int(bbox[0,0]),int(bbox[0,2]))
    #     print(int(bbox[0,1]),int(bbox[0,3]))
    #     z = torch.zeros(1500,3000)
    #     z[int(bbox[0,0]):int(bbox[0,2]),int(bbox[0,1]):int(bbox[0,3])] = 1
    #     from torchvision.utils import save_image
    #     save_image(z,'/home/junkyu/project/tumor_detection/densenet_RPN/data/black2/'+i)
    #     temp = img[int(bbox[0,0]):int(bbox[0,2]),int(bbox[0,1]):int(bbox[0,3])]
#######################################################################################
    # dir = os.listdir('/home/junkyu/project/tumor_detection/unet/newdata/img/')
    # for i in dir:
    #     img = cv2.imread('/home/junkyu/project/tumor_detection/unet/newdata/img/'+i)
    #     mask = cv2.imread('/home/junkyu/project/tumor_detection/unet/newdata/mask/'+i)
    #     ttt = cv2.bitwise_and(img,mask)
    #     cv2.imwrite('/home/junkyu/project/tumor_detection/unet/newdata/mask_img/'+i,ttt)
    dir = os.listdir('/home/junkyu/project/tumor_detection/unet/newdata/mask_img')
    for i in dir:
        img = Image.open('/home/junkyu/project/tumor_detection/unet/newdata/mask_img/'+i).resize((1500,750))
        img.save('/home/junkyu/project/tumor_detection/unet/newdata/mask_img2/'+i)
