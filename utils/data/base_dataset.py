'''
Author: error: git config user.name && git config user.email & please set dead value or install git
Date: 2023-01-09 22:14:10
LastEditors: error: git config user.name && git config user.email & please set dead value or install git
LastEditTime: 2023-01-10 12:01:02
FilePath: /Lingfeng He/xiongyali_new_idea/unsupervised_RGB_IR/cluster-contrast-reid原始代码/clustercontrast/utils/data/base_dataset.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
# encoding: utf-8
import numpy as np


class BaseDataset(object):
    """
    Base class of reid dataset
    """

    def get_imagedata_info(self, data):
        pids, cams = [], []
        for _, pid, camid in data:
            pids += [pid]
            cams += [camid]
        pids = set(pids)
        cams = set(cams)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        return num_pids, num_imgs, num_cams

    def print_dataset_statistics(self):
        raise NotImplementedError

    @property
    def images_dir(self):
        return None


class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """

    def print_dataset_statistics(self, train_rgb, train_ir, test_rgb, test_ir, query_rgb, gallery_rgb, query_ir, gallery_ir):
        num_train_rgb_pids, num_train_rgb_imgs, num_train_rgb_cams = self.get_imagedata_info(train_rgb)
        num_train_ir_pids, num_train_ir_imgs, num_train_ir_cams = self.get_imagedata_info(train_ir)
        num_test_rgb_pids, num_test_rgb_imgs, num_test_rgb_cams = self.get_imagedata_info(test_rgb)
        num_test_ir_pids, num_test_ir_imgs, num_test_ir_cams = self.get_imagedata_info(test_ir)
        num_query_rgb_pids, num_query_rgb_imgs, num_query_rgb_cams = self.get_imagedata_info(query_rgb)
        num_gallery_rgb_pids, num_gallery_rgb_imgs, num_gallery_rgb_cams = self.get_imagedata_info(gallery_rgb)
        num_query_ir_pids, num_query_ir_imgs, num_query_ir_cams = self.get_imagedata_info(query_ir)
        num_gallery_ir_pids, num_gallery_ir_imgs, num_gallery_ir_cams = self.get_imagedata_info(gallery_ir)
        

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print("   train-RGB   | {:5d} | {:8d} | {:9d}".format(num_train_rgb_pids, num_train_rgb_imgs, num_train_rgb_cams))
        print("   train-IR    | {:5d} | {:8d} | {:9d}".format(num_train_ir_pids, num_train_ir_imgs, num_train_ir_cams))
        print("   test-RGB    | {:5d} | {:8d} | {:9d}".format(num_test_rgb_pids, num_test_rgb_imgs, num_test_rgb_cams))
        print("    test-IR    | {:5d} | {:8d} | {:9d}".format(num_test_ir_pids, num_test_ir_imgs, num_test_ir_cams))
        print("   query-RGB   | {:5d} | {:8d} | {:9d}".format(num_query_rgb_pids, num_query_rgb_imgs, num_query_rgb_cams))
        print("  gallery-RGB  | {:5d} | {:8d} | {:9d}".format(num_gallery_rgb_pids, num_gallery_rgb_imgs, num_gallery_rgb_cams))
        print("    query-IR   | {:5d} | {:8d} | {:9d}".format(num_query_ir_pids, num_query_ir_imgs, num_query_ir_cams))
        print("   gallery-IR  | {:5d} | {:8d} | {:9d}".format(num_gallery_ir_pids, num_gallery_ir_imgs, num_gallery_ir_cams))
        print("  ----------------------------------------")
