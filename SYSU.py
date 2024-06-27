'''
Author: error: git config user.name && git config user.email & please set dead value or install git
Date: 2023-01-09 22:27:21
LastEditors: error: git config user.name && git config user.email & please set dead value or install git
LastEditTime: 2023-01-18 13:35:14
FilePath: /Lingfeng He/xiongyali_new_idea/unsupervised_RGB_IR/cluster-contrast-reid原始代码/clustercontrast/datasets/SYSU.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from __future__ import print_function, absolute_import
import os
import os.path as osp
from utils.data import BaseImageDataset

class SYSUMM01(BaseImageDataset):
    """
    Market1501
    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    dataset_dir = 'SYSU-MM01'

    def __init__(self, args, root, verbose=True):
        super(SYSUMM01, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self._check_before_run()

        train_rgb, file_rgb = self._process_dir(mode='RGB', is_trainset=True)
        train_ir, file_ir = self._process_dir(mode='IR', is_trainset=True)
        test_rgb = self._process_dir(mode='RGB', is_trainset=False)
        test_ir = self._process_dir(mode='IR', is_trainset=False)

        if verbose:
            print("=> SYSU-MM01  loaded")
            self.print_dataset_statistics(train_rgb, train_ir, test_rgb, test_ir, train_rgb, train_rgb, train_ir, train_ir)

        self.train_rgb = train_rgb
        self.train_ir = train_ir
        self.test_rgb = test_rgb
        self.test_ir = test_ir
        self.query_rgb = train_rgb
        self.gallery_rgb = train_rgb
        self.query_ir = train_ir
        self.gallery_ir = train_ir
        self.file_rgb = file_rgb
        self.file_ir = file_ir

        self.num_train_rgb_pids, self.num_train_rgb_imgs, self.num_train_rgb_cams = self.get_imagedata_info(train_rgb)
        self.num_train_ir_pids, self.num_train_ir_imgs, self.num_train_ir_cams = self.get_imagedata_info(train_ir)
        self.num_test_rgb_pids, self.num_test_rgb_imgs, self.num_test_rgb_cams = self.get_imagedata_info(test_rgb)
        self.num_test_ir_pids, self.num_test_ir_imgs, self.num_test_ir_cams = self.get_imagedata_info(test_ir)
        self.num_query_rgb_pids, self.num_query_rgb_imgs, self.num_query_rgb_cams = self.get_imagedata_info(self.query_rgb)
        self.num_gallery_rgb_pids, self.num_gallery_rgb_imgs, self.num_gallery_rgb_cams = self.get_imagedata_info(self.gallery_rgb)
        self.num_query_ir_pids, self.num_query_ir_imgs, self.num_query_ir_cams = self.get_imagedata_info(self.query_ir)
        self.num_gallery_ir_pids, self.num_gallery_ir_imgs, self.num_gallery_ir_cams = self.get_imagedata_info(self.gallery_ir)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))


    def _process_dir(self, mode, is_trainset=True):
        
        if mode == 'RGB':
            cameras = ["cam1", "cam2", "cam4", "cam5"]
        else:
            cameras = ["cam3", "cam6"]        
        
        ## trainset
        if is_trainset:
            file_path_train = os.path.join(self.dataset_dir, "exp/train_id.txt")
            file_path_val = os.path.join(self.dataset_dir, "exp/val_id.txt")
            with open(file_path_train, 'r') as file:
                ids = file.read().splitlines()
                ids = [int(y) for y in ids[0].split(',')]
                id_train = ["%04d" % x for x in ids]

            with open(file_path_val, 'r') as file:
                ids = file.read().splitlines()
                ids = [int(y) for y in ids[0].split(',')]
                id_val = ["%04d" % x for x in ids]

            # combine train and val split
            id_train.extend(id_val)
            pid_container = set()
            for pid in id_train:
                pid_container.add(pid)
            pid2label = {pid: label for label, pid in enumerate(pid_container)}
            
            dataset = []
            files = []
            for id in sorted(id_train):
                for cam in cameras:
                    img_dir = os.path.join(self.dataset_dir, cam, id)
                    if os.path.isdir(img_dir):
                        img_path = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
                        for path in img_path:
                            dataset.append((path, int(id), cam)) 
                            files.append(path)
                        
            return dataset, files    
        ## testset
        else:
            file_path_test = os.path.join(self.dataset_dir, "exp/test_id.txt")
            with open(file_path_test, 'r') as file:
                ids = file.read().splitlines()
                ids = [int(y) for y in ids[0].split(',')]
                id_all = ["%04d" % x for x in ids]
            
            dataset = []
            for id in sorted(id_all):
                for cam in cameras:
                    img_dir = os.path.join(self.dataset_dir, cam, id)
                    if os.path.isdir(img_dir):
                        img_path = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
                        for path in img_path:
                            dataset.append((path, int(id), cam))
            
            return dataset
                                            