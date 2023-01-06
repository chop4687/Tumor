import os
from torch.utils.data import Dataset
from PIL import Image
from sklearn.model_selection import KFold

class pretrain_sample(Dataset):
    def __init__(self, root, transform = None, train = None, CV = 1):
        self.root = root
        dir = os.listdir('/home/junkyu/data/1700_tumor_annotation/img')

        kf = KFold(n_splits=5)

        for i in range(int(CV)):
            temp = next(kf.split(dir[:int(len(dir)*0.85)]))

        if train is None:
            dir = [dir[i] for i in temp[0]]
        elif train == 'val':
            dir = [dir[i] for i in temp[1]]

        elif train == 'test':
            dir = dir[int(len(dir)*0.85):]

        self.data_point = dir
        self.transform = transform

    def _load(self, file):
        input_adress = self.root + '/img/' + file
        target_adress = self.root + '/ann2/' + file[:-4] + '.png'
        input = Image.open(input_adress).convert('L')
        target = Image.open(target_adress).convert('L')
        return input, target


    def __len__(self):
        return len(self.data_point)

    def __getitem__(self, idx):
        file = self.data_point[idx]
        x, y = self._load(file)

        if self.transform is not None:
            x, y = self.transform((x,y))

        return x, y

# class pretrain_sample(Dataset):
#     def __init__(self, root, transform = None, train = None, CV = 1):
#         self.root = root
#         dir = os.listdir('/home/junkyu/data/1700_tumor_annotation/add_DC/img')
#
#         # kf = KFold(n_splits=5)
#         #
#         # for i in range(int(CV)):
#         #     temp = next(kf.split(dir[:int(len(dir)*0.85)]))
#         #
#         # if train is None:
#         #     dir = [dir[i] for i in temp[0]]
#         # elif train == 'val':
#         #     dir = [dir[i] for i in temp[1]]
#         #
#         # elif train == 'test':
#         #     dir = dir[int(len(dir)*0.85):]
#         self.data_point = dir
#         self.transform = transform
#
#     def _load(self, file):
#         input_adress = self.root + '/img/' + file
#         #target_adress = self.root + '/ann2/' + file[:-4] + '.png'
#         input = Image.open(input_adress).convert('L')
#         #target = Image.open(target_adress).convert('L')
#         return input, file#, target
#
#
#     def __len__(self):
#         return len(self.data_point)
#
#     def __getitem__(self, idx):
#         file = self.data_point[idx]
#         x,name = self._load(file)
#
#         if self.transform is not None:
#             #x, y = self.transform((x,y))
#             x = self.transform(x)
#
#         return x, name#, y
