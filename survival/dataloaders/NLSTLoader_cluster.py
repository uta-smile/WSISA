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


def fm_assemble(feat_list, num_sample, which_feat='global_maxpool', random_sample=True, return_patch_coord=False):
    """ for use of 2048 logits in selected cluster
    assemble the feature maps for later processing. e.g.
    100 * [2048*1] --> 2048*1*100
    100 * one_sample --> one_bigger_sample
    """
    big_featmap = None
    if num_sample > 0 and len(feat_list) > num_sample:
        if random_sample:
            sample_feat = random.sample(feat_list, num_sample)
        else:
            random.seed(3)
            sample_feat = random.sample(feat_list, num_sample)
    else:
        sample_feat = feat_list
    # sample_feat = feat_list[:num_sample]
    for idx, feat_file in enumerate(sample_feat):
        # read data directly at beginning of each row (col 0)
        fn = sample_feat[idx]
        fn = os.path.join('/home/cy/ssd_4t/nlst/nlst_feat_swav',fn.split('/')[-2],fn.split('/')[-1])
        if idx == 0:
            big_featmap = np.load(fn, allow_pickle=True).item()[which_feat]
        # concat others behind the first one to form a row containing "col" items
        else:
            tmp_feat = np.load(fn, allow_pickle=True).item()[which_feat]
            big_featmap = np.concatenate((big_featmap, tmp_feat), axis=1)
    # the last row
    #big_featmap = np.concatenate((big_featmap, feat_map_row), axis=-2)
    # import pdb
    # pdb.set_trace()
    big_featmap = np.squeeze(big_featmap, axis=-1)
    assert big_featmap is not None, "featmap is None!"
    if return_patch_coord:
        all_coord = get_coord_from_fn(sample_feat)
        return big_featmap, all_coord
    else:
        return big_featmap, []

def fm_assemble0(feat_list, num_sample, which_feat='global_maxpool', random_sample=True, return_patch_coord=False):
    """ for use of embeddings in elected clusters
    assemble the feature maps for later processing. e.g.
    100 * [2048*1] --> 2048*1*100
    100 * one_sample --> one_bigger_sample
    """
    big_featmap = None
    if num_sample > 0 and len(feat_list) > num_sample:
        if random_sample:
            sample_feat = random.sample(feat_list, num_sample)
        else:
            random.seed(3)
            sample_feat = random.sample(feat_list, num_sample)
    else:
        return [], 5
        sample_feat = feat_list
    # sample_feat = feat_list[:num_sample]
    big_featmap=[]
    for idx, feat_file in enumerate(sample_feat):
        # read data directly at beginning of each row (col 0)
        # if idx == 0:
            # big_featmap = np.load(sample_feat[idx], allow_pickle=True).item()[which_feat]
        # concat others behind the first one to form a row containing "col" items
        # else:
            tmp_feat = np.load(sample_feat[idx], allow_pickle=True).item()[which_feat]
            tmp_feat = np.squeeze(tmp_feat)
            big_featmap.append(tmp_feat)
            # big_featmap = np.concatenate((big_featmap, tmp_feat), axis=1)
    # the last row
    #big_featmap = np.concatenate((big_featmap, feat_map_row), axis=-2)
    # import pdb
    # pdb.set_trace()
    big_featmap = np.array(big_featmap)
    # big_featmap = np.squeeze(big_featmap, axis=-1)
    assert big_featmap is not None, "featmap is None!"
    if return_patch_coord:
        all_coord = get_coord_from_fn(sample_feat)
        return big_featmap, all_coord
    else:
        return big_featmap, []


def index_dataset(list_dirs, data_dir, file_type='jpg', split='train'):
    """make index for each folder that containes N(1000) samples"""
    all_data = {}
    for dir in tqdm(list_dirs,
                    desc='Indexing [{}] dataset (list to dict)'.format(split),
                    dynamic_ncols=True):
        dir = str(dir)
        sub_dir = os.path.join(data_dir, dir)
        sub_list = get_file_list(sub_dir,
                                 fType=file_type,
                                 str_exclude='thumbnail')
        all_data[dir] = [os.path.join(dir, fn) for fn in sub_list]
    return all_data


def convert_index(inputpid, expandlabel):
    outputindex = {}
    for pid in inputpid:
        tmp = list(expandlabel[expandlabel['pid']==pid].img)
        outputindex[pid] = tmp
    # patient_num = [len(x) for x in outputindex]
    # outputindex = [y for x in outputindex for y in x]
    return outputindex


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
                 data_dir='/home/cy/ssd_4t/nlst/nlst_selected_cluster_plot/validpatients.csv',
                 label_file='/home/cy/ssd_4t/nlst/nlst_selected_cluster/selected_patch_cls20.csv',
                 split='train',
                 train_ratio=0.80,
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

        # with open(label_file, 'r') as fp:
        #     self.label_data = json.load(fp)

        label_path = data_dir
        # print(data_dir)
        # label_path = '/home/cy/ssd_4t/nlst/nlst_proto_cluster/embeddings_1000_cls5.csv'
        labels = pd.read_csv(label_path)
        temp = labels["status"]
        # expand_label_path = '/home/cy/ssd_4t/nlst/nlst_selected_cluster/patchsurv1000.csv'
        # expand_label_path = label_file
        # print(label_file)
        expand_label = pd.read_csv(label_file)
        # train/val=0.8/0.2 -> train/val/test=0.8/0.1/0.1
        train_split, test_split = split_train_test(temp.index, val_size=self.test_ratio)
        val_split, test_split = split_train_test(test_split, val_size=0.5)
        if split == 'train':
            self.split_dirs = labels['pid'].values[train_split]
            self.split_status = labels['status'].values[train_split]
            self.split_surv = labels['surv'].values[train_split]
        elif split == 'test':
            self.split_dirs = labels['pid'].values[test_split]
            self.split_status = labels['status'].values[test_split]
            self.split_surv = labels['surv'].values[test_split]
        elif split == 'val':
            self.split_dirs = labels['pid'].values[val_split]
            self.split_status = labels['status'].values[val_split]
            self.split_surv = labels['surv'].values[val_split]
        else:
            raise NameError('Wrong split name. Options: train / test / val')

        self.data_size = len(self.split_dirs)
        print('data size ', self.data_size)
        print(self.split_dirs)
        print(self.split_status)
        print(self.split_surv)
        self.dataset_dict = convert_index(self.split_dirs, expand_label)
        # load all data into memory
        if self.load_into_memory:
            print('load into memory')
            self.whole_dataset_in_memory, self.dataset_coord_in_memory = load_data_into_memory(
                self.split_dirs, self.dataset_dict, self.num_sample, self.feat_type, self.random_sample)


    def __getitem__(self, index):
        # first select WSI (a folder of 1000 patches)
        # print('dataloader index ', index)
        wsi_fn = self.split_dirs[index]
        # label = self.label_data[str(wsi_fn)]
        # import pdb; pdb.set_trace()
        # [1,2,3,4]->[0,1,2,3] make it start from 0
        """need to filter some NULL labels, try"""

        label_surv_time = float(self.split_surv[index])
        label_status = int(self.split_status[index])
        # label_stage = label['overall_stage']
        if self.load_into_memory:
            big_feat = self.whole_dataset_in_memory[index]
        else:
            big_feat, flag = fm_assemble0(self.dataset_dict[wsi_fn], self.num_sample, which_feat=self.feat_type)
            if flag == 5:
                index = index - 1 if index > 0 else index + 1 
                return self.__getitem__(index)
        big_feat = torch.FloatTensor(big_feat)
        return big_feat, torch.FloatTensor([label_surv_time]), torch.FloatTensor([label_status]), wsi_fn

    def __len__(self):
        return self.data_size



if __name__ == '__main__':
    data_dir = '/media/ssd_4t/TCGA/feature/resnet50'
