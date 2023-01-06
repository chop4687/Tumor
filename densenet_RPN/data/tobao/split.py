import os
import shutil

def dell(l):
    return l[:-4]
if __name__ == '__main__':
    root = '/home/junkyu/project/tumor_detection/densenet_RPN/data/tobao/'
    temp = os.listdir(root+'bbox_idx')
    temp = map(dell,temp)
    naaa = ['OKC']
    for n in naaa:
        tmp = os.listdir('/home/junkyu/data/all_tumor/'+n)
        tmp = list(map(dell,tmp))
        for i,t in enumerate(temp):
            if t in tmp:
                shutil.copy(root+'mask_img/'+t+'.png',root+'label_mask_img/'+n+'/'+t+'.png')
