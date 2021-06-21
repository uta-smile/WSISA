"""
Define pytorch dataloader for MIL datasets


"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import deepdish as dd
import scipy.ndimage as ndimage
import cv2

from sklearn.model_selection import train_test_split
dataset = 'NLST'
root_dir = "/home/jy/Data_JY/Code/Xinliang/patches/20x/{}/".format(dataset)
#GBM
# rgb_mean = np.array([195.44496056, 140.9117753 , 176.19348687], dtype=np.float32)
# rgb_std = np.array([48.47225281, 54.51432295, 47.76309123], dtype=np.float32)

#LUSC
# rgb_mean = np.array([180.24795886, 125.36680711, 159.26268276], dtype=np.float32)
# rgb_std = np.array([54.55753732, 61.06904195, 55.04639565], dtype=np.float32)

# NLST
# rgb_mean = np.array([159.97859971, 127.26877747, 152.02868696], dtype=np.float32)
# rgb_std = np.array([66.08328181, 69.80747985, 60.83659031], dtype=np.float32)
rgb_mean = np.array([177.57853624, 150.56494972, 170.63516087])
rgb_std = np.array([65.25112147, 75.04014587, 62.77260448])
#MCO
# rgb_mean = np.array([206.23609049, 150.09241683, 172.54615828], dtype=np.float32)
# rgb_std = np.array([50.90358102, 54.08344129, 55.70695798], dtype=np.float32)

class WSISA_get_feat_dataloader():
    def __init__(self, data_path, id_slice, batch_size):
        feat_dataset = WSISA_get_feat_dataset(data_path, id_slice,
                                              transform=transforms.Compose([ToTensor_GetFeat()]))
        self.dataloader  = DataLoader(feat_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    def get_loader(self):
        return self.dataloader




class WSISA_get_feat_dataset(Dataset):

    def __init__(self, list_path, id_slice, transform=None, train=True):
        """
        Give patch file path, corresponding surv time and status
        :param list_path:
        """
        self.list_path = list_path
        self.random = train
        self.transform = transform
        self.id_slice = id_slice

    def __len__(self):
        return len(self.list_path)

    def __getitem__(self, idx):

        img_path = self.list_path[idx]
        id_slice = self.id_slice[idx]

        # img = img_path.split('_thumbnail.jpeg')[0]
        # svs_name = img.split('_')[0]
        # full_path = os.path.join(root_dir, svs_name, img + '.jpg')

        img_data = cv2.imread(img_path)
        img_data = np.asarray(img_data, dtype=np.float64)
        img_data = (img_data - rgb_mean) / rgb_std
        img_data = np.swapaxes(img_data, 0, -1)

        sample = {'feat': img_data, 'slice': np.asarray(id_slice, dtype=np.float32)}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor_GetFeat(object):
    """Convert ndarrays in sample to Tensors."""

    # swap color axis because
    # numpy image: H x W x C
    # torch image: C X H X W
    def __call__(self, sample):
        image, idslice = sample['feat'], sample['slice']

        return {'feat': torch.FloatTensor(image), 'slice': torch.FloatTensor(idslice)}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    # swap color axis because
    # numpy image: H x W x C
    # torch image: C X H X W
    def __call__(self, sample):
        image, time, status = sample['feat'], sample['time'], sample['status']

        return {'feat': torch.from_numpy(image), 'time': torch.FloatTensor(time), 'status':torch.FloatTensor(status)}