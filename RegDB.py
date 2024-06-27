import os
import os.path as osp
from utils.data import BaseImageDataset

class RegDB(BaseImageDataset):
    
    dataset_dir = 'RegDB'
    
    def __init__(self, args, root, verbose=True):
        super(RegDB, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self._check_before_run()
        self.trial = args.regdb_trial
        
        train_rgb, file_rgb = self._process_dir(mode='RGB')
        train_ir, file_ir = self._process_dir(mode='IR')
        
        self.train_rgb = train_rgb
        self.train_ir = train_ir
        self.file_rgb = file_rgb
        self.file_ir = file_ir
        
        self.num_train_rgb_pids, self.num_train_rgb_imgs, self.num_train_rgb_cams = self.get_imagedata_info(train_rgb)
        self.num_train_ir_pids, self.num_train_ir_imgs, self.num_train_ir_cams = self.get_imagedata_info(train_ir)
        
        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print("   train-RGB   | {:5d} | {:8d} | {:9d}".format(self.num_train_rgb_pids, self.num_train_rgb_imgs, self.num_train_rgb_cams))
        print("   train-IR    | {:5d} | {:8d} | {:9d}".format(self.num_train_ir_pids, self.num_train_ir_imgs, self.num_train_ir_cams))
        print("  ----------------------------------------")

        
    def _process_dir(self, mode):
        if mode == 'RGB':
            train_list = os.path.join(self.dataset_dir, "idx/train_visible_{}".format(self.trial) + ".txt")
        elif mode == 'IR':
            train_list = os.path.join(self.dataset_dir, "idx/train_thermal_{}".format(self.trial) + ".txt")
        img_path, label = self.load_file_label(train_list)
        
        dataset = []
        file = []
        for i in range(len(img_path)):
            file_img = osp.join(self.dataset_dir, img_path[i])
            file.append(file_img)
            dataset.append((file_img, label[i], int(0)))
        return dataset, file
        
        
    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        
        
    def load_file_label(self, input_data_path):
        with open(input_data_path) as f:
            data_file_list = open(input_data_path, 'rt').read().splitlines()
            # Get full list of image and labels
            file_image = [s.split(' ')[0] for s in data_file_list]
            file_label = [int(s.split(' ')[1]) for s in data_file_list]

        return file_image, file_label
        
        