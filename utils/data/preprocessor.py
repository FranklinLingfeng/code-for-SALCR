'''
Author: error: git config user.name && git config user.email & please set dead value or install git
Date: 2023-01-09 22:14:10
LastEditors: error: git config user.name && git config user.email & please set dead value or install git
LastEditTime: 2023-01-11 10:43:28
FilePath: /Lingfeng He/xiongyali_new_idea/unsupervised_RGB_IR/cluster-contrast-reid原始代码/clustercontrast/utils/data/preprocessor.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from __future__ import absolute_import
import os
import os.path as osp
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import math
from PIL import Image


class Preprocessor(Dataset):
    def __init__(self, dataset, root=None, transform=None, to_grey=False):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform
        self.to_grey = to_grey

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)
            
        img = Image.open(fpath).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, fname, pid, camid, index
