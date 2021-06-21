import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms import Compose, Resize, RandomCrop, CenterCrop, RandomHorizontalFlip, ToPILImage, ToTensor, Normalize
from tqdm import tqdm
import numpy as np
import random
from sklearn.model_selection import train_test_split
import json


def get_file_list(fPath, fType, str_include='', str_exclude=' '):
    """Get the file list in the path `fPath` according to the given file type `fType`
    """
    if not isinstance(fType, str):
        print('File type should be string!')
        return
    else:
        return [
            os.path.join(root, f)
            for root, _, fl in tqdm(list(os.walk(fPath)),
                                    dynamic_ncols=True,
                                    ascii=False,
                                    desc='Indexing dataset') for f in fl
            if (f.endswith(fType) and str_include in f and str_exclude not in f
                )
        ]


def get_feat_list(fPath, fType):
    """Get the file list in the path `fPath` according to the given file type `fType`
    """
    if not isinstance(fType, str):
        print('File type should be string!')
        return
    else:
        return [
            os.path.join(root, f) for root, _, fl in list(os.walk(fPath))
            for f in fl if (f.endswith(fType))
        ]


def split_train_test(data_list, val_size=0.2):
    """Use twice to 0.6/0.2/0.2 train/val/test split"""
    data_list.sort()  # sort to make sure totally same split
    train_split, test_split = train_test_split(data_list,
                                               test_size=val_size,
                                               random_state=42)
    return train_split, test_split


def fm_assemble(feat_list, num_sample):
    """assemble the feature maps for later processing. e.g.
    100 * [2048*7*7] --> 2048*70*70
    100 * one_sample --> one_bigger_sample """
    big_featmap = None
    sample_feat = random.sample(feat_list, num_sample)
    row = col = int(np.sqrt(num_sample))  # assemble
    for idx, feat_file in enumerate(sample_feat):
        # read data directly at beginning of each row (col 0)
        if idx % col == 0:
            # save 0th feat_map_row before restart at 1st row
            if idx // row == 1:  #  feat_data = None if idx // row == 0:
                big_featmap = feat_map_row
            elif idx // row > 1:
                big_featmap = np.concatenate((big_featmap, feat_map_row),
                                             axis=-2)
            feat_map_row = np.load(sample_feat[idx],
                                   allow_pickle=True)['arr_0'].item()['layer4']
        # concat others behind the first one to form a row containing "col" items
        elif idx % col != 0:
            tmp_feat = np.load(sample_feat[idx],
                               allow_pickle=True)['arr_0' \
                                                  ''].item()['layer4']
            feat_map_row = np.concatenate((feat_map_row, tmp_feat), axis=-1)
    # the last row
    big_featmap = np.concatenate((big_featmap, feat_map_row), axis=-2)
    assert big_featmap is not None, "featmap is None!"
    return big_featmap


def index_dataset(list_dirs, data_dir, split='train'):
    """make index for each folder that containes N(1000) samples"""
    all_data = {}
    for dir in tqdm(list_dirs,
                    desc='Indexing [{}] dataset (list to dict)'.format(split),
                    dynamic_ncols=True):
        sub_dir = os.path.join(data_dir, dir)
        sub_list = get_feat_list(sub_dir, fType='npz')
        all_data[dir] = [os.path.join(dir, fn) for fn in sub_list]
    return all_data


# add @func in the future to split
class PatchFeature(Dataset):
    """
    Sample `num_sample` patches from total 1000 patches of one WSI.
    Each folder contains 1000 patches of one WSI.
    """
    def __init__(self,
                 num_sample=100,
                 data_dir='/media/titan_3t/mco/mco_patch_feature/feature1000',
                 label_file='/media/ssd/all_mco_with_label.json',
                 split='train',
                 train_ratio=0.6):
        self.test_ratio = 1 - train_ratio
        self.num_sample = num_sample
        assert os.path.exists(label_file), "Label file not exist!"
        with open(label_file, 'r') as fp:
            self.label_data = json.load(fp)

        self.data_dir = data_dir
        self.all_dirs = os.listdir(self.data_dir)
        # self.all_dirs = [
        #     dir for dir in self.all_dirs
        #     if dir not in self.label_data['patch1k_no_label']
        # ]
        # train/val=0.8/0.2 -> train/val/test=0.8/0.1/0.1
        train_split, test_split = split_train_test(self.all_dirs,
                                                   val_size=self.test_ratio)
        val_split, test_split = split_train_test(test_split, val_size=0.5)
        if split == 'train':
            self.split_dirs = train_split
        elif split == 'test':
            self.split_dirs = test_split
        elif split == 'val':
            self.split_dirs = val_split
        else:
            raise NameError('Wrong split name. Options: train / test / val')

        self.data_size = len(self.split_dirs)
        self.dataset_dict = index_dataset(self.split_dirs,
                                          self.data_dir,
                                          split=split)

    def __getitem__(self, index):
        # first select WSI (a folder of 1000 patches)
        wsi_fn = self.split_dirs[index]
        label = self.label_data[wsi_fn.split('.')[0]]
        # [1,2,3,4]->[0,1,2,3] make it start from 0
        """need to filter some NULL labels, try"""
        label_tumor = label['primary_tumor']
        # label_stage = label['overall_stage']
        big_feat = fm_assemble(self.dataset_dict[wsi_fn], self.num_sample)
        return big_feat, torch.FloatTensor([label_tumor]), wsi_fn

    def __len__(self):
        return self.data_size
        # return 10


if __name__ == '__main__':
    # unit_test1 feature folder func
    data_dir = '/media/titan_3t/mco/mco_patch_feature/feature1000'
    # groups = os.listdir(data_dir)
    # print(len(groups))
    # dataset_dict = index_dataset(groups, data_dir)
    # # print(dataset_dict)
    # for k, v in dataset_dict.items():
    #     if len(v) != 1000:
    #         print(k)
    #         print(len(v))
    #
    # tmp_key = list(dataset_dict.keys())[0]
    # big_featmap = fm_assemble(dataset_dict[tmp_key], num_sample=100)
    # print(big_featmap.shape)

    # 2nd unit test
    dataset_train = PatchFeature(num_sample=100, split='train')
    print(dataset_train.__len__())
    dataloader_feature = DataLoader(dataset=dataset_train,
                                    batch_size=2,
                                    num_workers=2,
                                    shuffle=False,
                                    pin_memory=True)
    for idx, (feat, label, fn) in tqdm(enumerate(dataloader_feature),
                                       total=dataloader_feature.__len__()):
        # if idx > 5:
        #     break
        # print(feat.shape)
        print(fn, 'label:{}'.format(label))

    # dataset_test = PatchFeature(num_sample=100, split='test')
    # print(dataset_test.__len__())
    # dataloader_feature = DataLoader(dataset=dataset_test,
    #                                 batch_size=2,
    #                                 num_workers=2,
    #                                 shuffle=False,
    #                                 pin_memory=True)
    # for idx, (feat, label, fn) in tqdm(enumerate(dataloader_feature),
    #                                    total=dataset_test.__len__()):
    #     if idx > 5:
    #         break
    #     print(feat.shape)
    #     print(fn, 'label:{}'.format(label))
