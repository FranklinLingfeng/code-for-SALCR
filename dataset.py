'''
Author: error: git config user.name && git config user.email & please set dead value or install git
Date: 2023-01-16 14:38:24
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2023-02-04 15:11:08
FilePath: /Lingfeng He/xiongyali_new_idea/assignment/dataset.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import numpy as np
import os.path as osp
import pickle
import torchvision.transforms as transforms
from ChannelAug import ChannelAdap, ChannelAdapGray, ChannelRandomErasing, LinearTransform
import random
import torch.utils.data as data
from PIL import Image
import math
import torch
from numpy.random import beta


class ToGrey(object):
    """ Adaptive selects a channel or two channels.
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value. 
    """

    def __init__(self, gray=3):
        self.gray = gray

    def __call__(self, img):

        tmp_img = 0.2989 * img[0, :, :] + 0.5870 * img[1, :, :] + 0.1140 * img[2, :, :]
        img[0, :, :] = tmp_img
        img[1, :, :] = tmp_img
        img[2, :, :] = tmp_img
        return img


class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) >= self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img


class ChannelExchange(object):
    """ Adaptive selects a channel or two channels.
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value. 
    """

    def __init__(self, gray=3, probability=1):
        self.gray = gray
        self.probability = probability

    def __call__(self, img):

        idx = random.randint(0, self.gray)

        if idx == 0:
            # random select R Channel
            img[1, :, :] = img[0, :, :]
            img[2, :, :] = img[0, :, :]
        elif idx == 1:
            # random select B Channel
            img[0, :, :] = img[1, :, :]
            img[2, :, :] = img[1, :, :]
        elif idx == 2:
            # random select G Channel
            img[0, :, :] = img[2, :, :]
            img[1, :, :] = img[2, :, :]
            
        else:
            if random.uniform(0, 1) > self.probability:
                # return img
                img = img
            else:
                tmp_img = 0.2989 * img[0, :, :] + 0.5870 * img[1, :, :] + 0.1140 * img[2, :, :]
                img[0, :, :] = tmp_img
                img[1, :, :] = tmp_img
                img[2, :, :] = tmp_img
        return img
    
    
class togrey(object):
    
    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, img):     
        
        tmp_img = 0.2989 * img[0, :, :] + 0.5870 * img[1, :, :] + 0.1140 * img[2, :, :]
        img[0, :, :] = tmp_img
        img[1, :, :] = tmp_img
        img[2, :, :] = tmp_img   
        
        return img
    
class ToRGB(object):
    
    def __init__(self, probability=1.0):
        self.probability = probability

    def __call__(self, img):     
        
        H, W = img.shape[1], img.shape[2]
        tmp_img = torch.zeros(3, H, W)
        tmp_img[0, :, :] = img[0, :, :]
        tmp_img[1, :, :] = img[0, :, :]
        tmp_img[2, :, :] = img[0, :, :]   
        
        return img
    
    
class GammaTransform(object):

    def __init__(self, probability=0.5):
        self.probability = probability
        self.gamma_small = 0.5
        self.gamma_big = 3.0

    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            img = img

        else:
            gamma = random.uniform(self.gamma_small, self.gamma_big)
            img = img ** gamma

        return img


class WeightedGray(object):

    def __init__(self, probability=0.5):
        self.probability = probability
    
    def __call__(self, img):
        if random.uniform(0, 1) > self.probability:
            img = img
        else:
            alpha = random.uniform(0, 1)
            alpha1 = random.uniform(0, alpha)
            alpha2 = alpha - alpha1
            tmp_img = alpha1 * img[0, :, :] + alpha2 * img[1, :, :] + (1 - alpha1 - alpha2) * img[2, :, :]
            img[0, :, :] = tmp_img
            img[1, :, :] = tmp_img
            img[2, :, :] = tmp_img

        return img  
    

class ChannelCutmix(object):

    def __init__(self, probability=0.5, sl = 0.02, sh = 1.0, r1 = 0.3):
        self.probability = probability
        self.gray = 2
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):
        if random.uniform(0, 1) > self.probability:
            return img

        else:
            bg_idx, fg_idx = random.sample(range(3), 2)
            # print(bg_idx, fg_idx)
            x_bg = torch.zeros_like(img)
            if bg_idx == 0:
                x_bg[0, :, :] = img[0, :, :]
            elif bg_idx == 1:
                x_bg[1, :, :] = img[1, :, :]
            elif bg_idx == 2:
                x_bg[2, :, :] = img[2, :, :]

            for attempt in range(100):

                area = img.size()[1] * img.size()[2]
        
                target_area = random.uniform(self.sl, self.sh) * area
                aspect_ratio = random.uniform(self.r1, 1/self.r1)

                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))

                if w < img.size()[2] and h < img.size()[1]:
                    x1 = random.randint(0, img.size()[1] - h)
                    y1 = random.randint(0, img.size()[2] - w)

                    if fg_idx == 0:
                        x_bg[0, x1:x1+h, y1:y1+w] = img[0, x1:x1+h, y1:y1+w]
                    elif fg_idx == 1:
                        x_bg[1, x1:x1+h, y1:y1+w] = img[1, x1:x1+h, y1:y1+w]
                    elif fg_idx == 2:
                        x_bg[2, x1:x1+h, y1:y1+w] = img[2, x1:x1+h, y1:y1+w]
                    break
                
            return x_bg




class SpectrumJitter(object):

    def __init__(self, probability=0.5):
        self.probability = probability
        self.gray = 2

    def __call__(self, img):
        if random.uniform(0, 1) > self.probability:
            img = img

        else:
            idx = random.randint(0, self.gray)
            # print(idx)
            x_ds = torch.zeros_like(img)
            if idx == 0:
                # random select R Channel
                x_ds[0, :, :] = img[0, :, :]
                x_ds[1, :, :] = 0
                x_ds[2, :, :] = 0
            elif idx == 1:
                # random select B Channel
                x_ds[0, :, :] = 0
                x_ds[1, :, :] = img[1, :, :]
                x_ds[2, :, :] = 0
            elif idx == 2:
                # random select G Channel
                x_ds[0, :, :] = 0
                x_ds[1, :, :] = 0
                x_ds[2, :, :] = img[2, :, :]

            beta = random.uniform(0, 1)
            # img = x_ds
            img = beta * img + (1 - beta) * x_ds

        return img


## train set
class pseudo_label_dataset(data.Dataset):
    
    def __init__(self, args, RGB_set, IR_set, RGB_instance_IR_label=None, IR_instance_RGB_label=None,
                 colorIndex = None, thermalIndex = None, 
                 img_h=288, img_w=144, epoch=0, stage=None):
        
        self.epoch=epoch

        self.file_IR = []
        self.label_IR = []
        self.label_IR_ = None
        self.file_RGB = []
        self.label_RGB = []  
        self.label_RGB_ = None    
    
        for fname, label in RGB_set:
            self.file_RGB.append(fname)
            self.label_RGB.append(label)
        for fname, label in IR_set:
            self.file_IR.append(fname)
            self.label_IR.append(label)
        
        ## image transforms
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        ## stage one (DCL) augmentations
        ################################
        self.transform_stage_one = transforms.Compose( [
            transforms.ToPILImage(),
            transforms.Resize((img_h, img_w)),
            transforms.Pad(10),
            transforms.RandomCrop((img_h, img_w)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            LinearTransform(probability = 0.5),
            normalize,
            ChannelRandomErasing(probability = 0.5)])
            
        self.transform_ca_stage_one = transforms.Compose( [
            transforms.ToPILImage(),
            transforms.Resize((img_h, img_w)),
            transforms.Pad(10),
            transforms.RandomCrop((img_h, img_w)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.5,contrast=0.5,saturation=0.5,hue=0.5), 
            transforms.ToTensor(),
            normalize,
            ChannelRandomErasing(probability = 0.5),
            ChannelExchange(gray=2)])
        
        self.transform_thermal_stage_one = transforms.Compose( [
            transforms.ToPILImage(),
            transforms.Resize((img_h, img_w)),
            transforms.Pad(10),
            transforms.RandomCrop((img_h, img_w)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.5,contrast=0.5,saturation=0.5,hue=0.5),
            transforms.ToTensor(),
            normalize,
            ChannelRandomErasing(probability = 0.5),
            ChannelAdapGray(probability=0.5)])
        
        ## stage two augmentations
        ##########################
        self.transform_stage_two = transforms.Compose( [
            transforms.ToPILImage(),
            transforms.Resize((img_h, img_w)),
            transforms.Pad(10),
            transforms.RandomCrop((img_h, img_w)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            normalize,
            ChannelRandomErasing(probability = 0.5)])

        self.transform_ca_stage_two = transforms.Compose( [
            transforms.ToPILImage(),
            transforms.Resize((img_h, img_w)),
            transforms.Pad(10),
            transforms.RandomCrop((img_h, img_w)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.5,contrast=0.5,saturation=0.5,hue=0.5), 
            transforms.ToTensor(),
            normalize,
            ChannelRandomErasing(probability = 0.5),
            ChannelExchange(gray=2)])
        
        self.transform_thermal_stage_two = transforms.Compose( [
            transforms.ToPILImage(),
            transforms.Resize((img_h, img_w)),
            transforms.Pad(10),
            transforms.RandomCrop((img_h, img_w)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.5,contrast=0.5,saturation=0.5,hue=0.5), 
            transforms.ToTensor(),
            normalize,
            ChannelRandomErasing(probability = 0.5),
            ChannelAdapGray(probability=0.5)])
        
        self.cIndex = colorIndex
        self.tIndex = thermalIndex   
        
        self.y_rgb = None
        self.y_ir = None
        self.y_rgb_cm = None
        self.y_ir_cm = None
        self.y_rgb_p = None
        self.y_ir_p = None
        self.y_rgb_cm_p = None
        self.y_ir_cm_p = None

        self.mask_v = None
        self.mask_r = None

        self.cm_label_RGB = RGB_instance_IR_label
        self.cm_label_IR = IR_instance_RGB_label
        self.cm_label_RGB_ = None
        self.cm_label_IR_ = None
        
        self.stage = stage            
        
    def __getitem__(self, index):
        
        img1 = np.array(Image.open(self.file_RGB[self.cIndex[index]])) 
        img2 = np.array(Image.open(self.file_IR[self.tIndex[index]])) 

        img1_label = self.label_RGB[self.cIndex[index]]
        img2_label = self.label_IR[self.tIndex[index]]

        if self.stage == 'single':

            img1_0 = self.transform_stage_one(img1)
            img1_1 = self.transform_ca_stage_one(img1) 
            img2 = self.transform_thermal_stage_one(img2)            

            return img1_0, img1_1, img2, img1_label, img2_label, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

        elif self.stage == 'cross':

            img1_0 = self.transform_stage_two(img1)
            img1_1 = self.transform_ca_stage_two(img1) 
            img2 = self.transform_thermal_stage_two(img2)

            y1 = self.y_rgb[self.cIndex[index]]
            y2 = self.y_ir[self.tIndex[index]]
        
            img1_cross_label = self.cm_label_RGB[self.cIndex[index]]
            img2_cross_label = self.cm_label_IR[self.tIndex[index]]
            cm_y1 = self.y_rgb_cm[self.cIndex[index]]
            cm_y2 = self.y_ir_cm[self.tIndex[index]]

            y1_p = self.y_rgb_p[self.cIndex[index]]
            y2_p = self.y_ir_p[self.tIndex[index]]
            cm_y1_p = self.y_rgb_cm_p[self.cIndex[index]]
            cm_y2_p = self.y_ir_cm_p[self.tIndex[index]]

            idx1 = self.cIndex[index]
            idx2 = self.tIndex[index]

            return img1_0, img1_1, img2, \
                img1_label, img2_label, img1_cross_label, img2_cross_label, \
                y1, y2, cm_y1, cm_y2, y1_p, y2_p, cm_y1_p, cm_y2_p, idx1, idx2
      
        else:
            print('Invalid training stage...')
        
    def _build_dataset(self, data_path):
        files = []
        labels = []
        num_outliers = 0
        if not osp.exists(data_path):
            raise RuntimeError("'{}' is not available".format(data_path))  
        with open(data_path, 'rb') as f:
            pseudo_label = pickle.load(f)
            
        for file in pseudo_label.keys():
            label = int(pseudo_label[file])
            if label != -1:
                files.append(file)
                labels.append(label)
            else:
                num_outliers += 1
        
        num_clusters = len(np.unique(labels))
        print('length of dataset:{:5d}\nnumber of clusters:{:5d}\nnumber of outliers:{:5d}'
              .format(int(len(files)), int(num_clusters), int(num_outliers)))
        
        return files, labels
    
    
class dataset_for_feature_extractor(data.Dataset):
    
    def __init__(self, trainset, gray=False, img_h=288, img_w=144):   
        
        self.trainset = trainset
        self.gray = gray
        
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform_test = transforms.Compose( [
            transforms.ToPILImage(),
            transforms.Resize((img_h, img_w)),
            transforms.ToTensor(),
            normalize])
        
        self.transform_gray = transforms.Compose( [
            transforms.ToPILImage(),
            transforms.Resize((img_h, img_w)),
            transforms.RandomGrayscale(p=1.0),
            transforms.ToTensor(),
            normalize])
        
    def __getitem__(self, index):
        img = np.array(Image.open(self.trainset[index][0]))
        if self.gray==True:
            img = self.transform_gray(img)
        else:
            img = self.transform_test(img)
        label = int(self.trainset[index][1])
        img_path = self.trainset[index][0]

        return img_path, img, label

    def __len__(self):
        return len(self.trainset)
    
    
class grey_dataset_for_feature_extractor(data.Dataset):
    
    def __init__(self, trainset, img_h=288, img_w=144):   
        
        self.trainset = trainset
        
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform_test = transforms.Compose( [
            transforms.ToPILImage(),
            transforms.Resize((img_h, img_w)),
            transforms.ToTensor(),
            normalize,
            ToGrey()])
        
    def __getitem__(self, index):
        img = np.array(Image.open(self.trainset[index][0]))
        img = self.transform_test(img)
        label = int(self.trainset[index][1])
        img_path = self.trainset[index][0]

        return img_path, img, label

    def __len__(self):
        return len(self.trainset)
    

        
        
## test set
class TestData(data.Dataset):
    def __init__(self, test_img_file, test_label, img_h=288, img_w=144):
        
        self.test_img_file = test_img_file
        self.test_label = test_label
        
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose( [
            transforms.ToPILImage(),
            transforms.Resize((img_h, img_w)),
            transforms.ToTensor(),
            normalize])

    def __getitem__(self, index):
        img1,  target1 = np.array(Image.open(self.test_img_file[index])),  self.test_label[index]
        img1 = self.transform(img1)
        return img1, target1

    def __len__(self):
        return len(self.test_img_file)