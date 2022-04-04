import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, RandomCrop, CenterCrop, RandomHorizontalFlip, ToPILImage, ToTensor, Normalize
from tqdm import tqdm
import numpy as np
import random
from sklearn.model_selection import train_test_split
import json
from PIL import Image
import random
import pandas as pd

random_num = 123
# random_num = 42


def get_file_list(fPath, fType, str_include='', str_exclude=' '):
    """Get the file list in the path `fPath` according to the given file type `fType`
    """
    if not isinstance(fType, str):
        print('File type should be string!')
        return
    else:
        return [
            os.path.join(root, f)
            for root, _, fl in list(os.walk(fPath))
            for f in fl
            if (f.endswith(fType) and str_include in f and str_exclude not in f)
        ]


def split_train_test(data_list, val_size=0.2):
    """Use twice to 0.6/0.2/0.2 train/val/test split"""
    # data_list.sort()  # sort to make sure totally same split
    # 之前random=3
    train_split, test_split = train_test_split(data_list, test_size=val_size, random_state=random_num)
    return train_split, test_split


def load_data_into_memory(wsi_fn_list,
                          wsi_patch_dir_list,
                          num_sample,
                          feat_type,
                          random_sample,
                          return_patch_coord=False):
    whole_dataset = []
    all_wsi_patch_coord = []
    for fn in tqdm(wsi_fn_list, desc='Loading all data into memory'):
        one_wsi_patches = wsi_patch_dir_list[fn]
        big_feat, all_coord = fm_assemble(one_wsi_patches,
                                          num_sample,
                                          feat_type,
                                          random_sample=random_sample,
                                          return_patch_coord=return_patch_coord)
        all_wsi_patch_coord.append(all_coord)
        whole_dataset.append(big_feat)

    return whole_dataset, all_wsi_patch_coord


def get_coord_from_fn(fl):
    """Get patch coordinates from file list"""
    all_coord = []
    for fn in fl:
        bn = os.path.basename(fn)
        x, y = bn.split('_')[1:3]
        all_coord.append([int(x), int(y)])
    return all_coord


# add @func in the future to split
class PatchFeature(Dataset):
    """
    Sample `num_sample` patches from total 1000 patches of one WSI.
    Each folder contains 1000 patches of one WSI.

    self.transforms = Compose([
        # ToPILImage(),
        Resize((self.img_resize, self.img_resize), Image.BILINEAR),
        ToTensor(),
        Normalize(mean=[0.82797706, 0.64284584, 0.71804111],
                  std=[[0.17938925,0.21496013,0.20454665]])  # MCO mean & std
    ])
    """

    def __init__(self,
                 num_sample=100,
                 data_dir='/media/ssd_4t/TCGA/feature/resnet50',
                 label_file='/media/ssd/all_mco_with_label.json',
                 split='train',
                 train_ratio=0.8036,
                 feat_type='global_maxpool',
                 load_into_memory=False,
                 random_sample=True,
                 patch_area='fg',
                 return_patch_coord=False):
        self.return_patch_coord = return_patch_coord
        self.load_into_memory = load_into_memory
        self.feat_type = feat_type
        self.random_sample = random_sample
        self.patch_area = patch_area

        self.test_ratio = 1 - train_ratio
        self.num_sample = num_sample

        features = 'features'
        fold = '2'
        if split == 'train':
            fn = '/home/cy/project/WSISA/results/train_patient_NLST_{}_fold{}_c10.csv'.format(features, fold)
        elif split == 'test':
            fn = '/home/cy/project/WSISA/results/test_patient_NLST_{}_fold{}_c10.csv'.format(features, fold)
        elif split == 'val':
            fn = '/home/cy/project/WSISA/results/validation_patient_NLST_{}_fold{}_c10.csv'.format(features, fold)
        else:
            raise NameError('Wrong split name. Options: train / test / val')

        self.labels = pd.read_csv(fn)
        self.split_dirs = self.labels['pid'].values
        self.split_status = self.labels['status'].values
        self.split_surv = self.labels['surv'].values

        self.labels = self.labels.fillna(0)
        self.data_size = self.labels.shape[0]
        print('data size ', self.data_size)

    def __getitem__(self, index):
        cur_feat = self.labels.iloc[index, 3:].values
        surv = self.split_surv[index]
        status = self.split_status[index]
        pid = self.split_dirs[index]

        return torch.FloatTensor(cur_feat), torch.FloatTensor([surv]), torch.FloatTensor([status]), pid

    def __len__(self):
        return self.data_size



if __name__ == '__main__':
    data_dir='/media/ssd_4t/TCGA/feature/resnet50'
