import cv2
import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.utils import save_image
if __name__ == '__main__':
    dir = os.listdir('/home/junkyu/project/tumor_detection/densenet_RPN/data/tumor_segmentation/')
    small = 1111110
    big = 0
    for i in dir:
        img = Image.open('/home/junkyu/project/tumor_detection/densenet_RPN/data/tumor_segmentation/' + i).convert('L')
        ten_img = T.ToTensor()(img)
        if ten_img.sum() < small:
            sm_name = i
            small = ten_img.sum()

        if ten_img.sum() > big:
            big_name = i
            big = ten_img.sum()
        if ten_img.sum() == 37436:
            print(i)
    print(sm_name, big_name)
