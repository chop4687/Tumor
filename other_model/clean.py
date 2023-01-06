import torch
import os
import shutil
if __name__ == '__main__':
    root = '/home/junkyu/project/tumor_detection/other_model/data/'
    for i,k in zip(['Ameloblastoma', 'dentigerous_cyst', 'OKC'], ['AB','DC','OKC']):
        temp = os.listdir(root+i)
        for j in temp:
            if j[:-4]+'.png' in os.listdir(root+'mask/tumor/'):
                shutil.copy(root+'mask/tumor/'+j[:-4]+'.png', root+'new_mask/'+k+'/'+j[:-4]+'.png')
