'''
Author: error: git config user.name && git config user.email & please set dead value or install git
Date: 2023-01-16 14:03:09
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2023-02-06 00:00:50
FilePath: /Lingfeng He/xiongyali_new_idea/assignment/main_train.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import argparse
import sys
import os.path as osp
import time
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from sklearn.cluster import DBSCAN
from sklearn import metrics
from cross_association import part_ot, knn_graph, agg_feature


import random

import collections
from trainer import AssignTrainer
from sampler import StageOneSampler, StageThreeSampler
from memory import ClusterMemory, WeightedMemory

import torch
from torch import nn
from torch.backends import cudnn
import torch.nn.functional as F
from utils.logging import Logger
from utils.serialization import save_checkpoint, copy_state_dict, load_checkpoint
from model.network import BaseResNet, SAFL, weights_init_kaiming
from optimizer import select_optimizer
from utils.faiss_rerank import compute_jaccard_distance
from SYSU import SYSUMM01
from RegDB import RegDB

from dataset import pseudo_label_dataset, TestData, dataset_for_feature_extractor
from data_manager import *
from evaluator import extract_features_for_cluster, test


def main():
    args = parser.parse_args()

    if args.seed is not None:

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

    main_worker(args)
    # for trial in [6, 7, 8]:
    #     args.regdb_trial = trial
    #     args.device = 2
    #     working_dir = osp.dirname(osp.abspath(__file__))
    #     args.logs_dir = osp.join(working_dir, 'logs/regdb_trial_'+str(trial))
    #     main_worker(args)

    # for trial in [9, 10]:
    #     args.regdb_trial = trial
    #     args.device = 0
    #     working_dir = osp.dirname(osp.abspath(__file__))
    #     args.logs_dir = osp.join(working_dir, 'logs/regdb_trial_'+str(trial))
    #     main_worker(args)
    
    
def GenIdx(train_color_label, train_thermal_label):
    color_pos = collections.defaultdict(list)
    for i in range(len(train_color_label)):
        color_pos[int(train_color_label[i].item())].append(i)
        
    thermal_pos = collections.defaultdict(list)
    for i in range(len(train_thermal_label)):
        thermal_pos[int(train_thermal_label[i].item())].append(i)
    
    return color_pos, thermal_pos


def generate_center(pseudo_labels, features):
    pseudo_labels = np.asarray(pseudo_labels)
    centers = collections.defaultdict(list)
    num_outliers = 0
    for i, label in enumerate(pseudo_labels):
        if label == -1:
            num_outliers += 1
            continue
        centers[pseudo_labels[i]].append(features[i])
    centers = [torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())] 
    centers = torch.stack(centers, dim=0)
    print('----number of outliers: {:5d}'.format(num_outliers))

    return centers



## return dbscan centers
def dbscan_cluster(args, features, cluster):

    rerank_dist = compute_jaccard_distance(features, k1=args.k1, k2=args.k2)
    print('----start DBScan clustering------')
    pseudo_labels = cluster.fit_predict(rerank_dist)
    print('Num clusters: {}'.format(pseudo_labels.max()+1))
    del rerank_dist
        
    return torch.tensor(pseudo_labels)



def print_intra_acc(RGB_true_label, IR_true_label, labels_RGB, labels_IR):

    N_rgb = labels_RGB.shape[0]
    N_ir = labels_IR.shape[0]
    
    RGB_mask_in = RGB_true_label.expand(N_rgb, N_rgb).eq(RGB_true_label.expand(N_rgb, N_rgb).t())
    IR_mask_in = IR_true_label.expand(N_ir, N_ir).eq(IR_true_label.expand(N_ir, N_ir).t())
    
    RGB_mask_in_p = labels_RGB.expand(N_rgb, N_rgb).eq(labels_RGB.expand(N_rgb, N_rgb).t())
    IR_mask_in_p = labels_IR.expand(N_ir, N_ir).eq(labels_IR.expand(N_ir, N_ir).t())
    
    RGB_in_acc = (RGB_mask_in_p * RGB_mask_in).sum() / RGB_mask_in.sum()
    IR_in_acc = (IR_mask_in_p * IR_mask_in).sum() / IR_mask_in.sum()
    RGB_in_recall = (RGB_mask_in_p * RGB_mask_in).sum() / RGB_mask_in_p.sum()
    IR_in_recall = (IR_mask_in_p * IR_mask_in).sum() / IR_mask_in_p.sum()
    
    print('RGB_in_recall:{:.4f} // IR_in_recall:{:.4f} // RGB_in_acc:{:.4f} // IR_in_acc:{:.4f}'
                    .format(RGB_in_acc, IR_in_acc, RGB_in_recall, IR_in_recall))


def print_cm_acc(RGB_true_label, IR_true_label, labels_RGB, labels_IR, RGB_instance_IR_label, IR_instance_RGB_label):

    N_rgb = labels_RGB.shape[0]
    N_ir = labels_IR.shape[0]
    
    RGB_mask_c_p = labels_RGB.expand(N_ir, N_rgb).t().eq(IR_instance_RGB_label.expand(N_rgb, N_ir))
    IR_mask_c_p = RGB_instance_IR_label.expand(N_ir, N_rgb).t().eq(labels_IR.expand(N_rgb, N_ir))
    true_mask_c = RGB_true_label.expand(N_ir, N_rgb).t().eq(IR_true_label.expand(N_rgb, N_ir))
    
    RGB_acc = (RGB_mask_c_p * true_mask_c).sum() / true_mask_c.sum()
    IR_acc = (IR_mask_c_p * true_mask_c).sum() / true_mask_c.sum()
    RGB_recall = (RGB_mask_c_p * true_mask_c).sum() / RGB_mask_c_p.sum()
    IR_recall = (IR_mask_c_p * true_mask_c).sum() / IR_mask_c_p.sum()

    print('RGB_recall:{:.4f} // IR_recall:{:.4f} // RGB_acc:{:.4f} // IR_acc:{:.4f}'
                    .format(RGB_acc, IR_acc, RGB_recall, IR_recall))


    
def main_worker(args):
    
    # cudnn.benchmark = True
    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    print("==========\nArgs:{}\n==========".format(args)) 
      
    ## build dataset
    end = time.time()
    print("============load data==========")
    if args.dataset == 'sysu':
        data_dir = osp.join(args.data_dir, 'SYSU-MM01')
        dataset = SYSUMM01(args, args.data_dir)     
           
        query_img1, query_label1, query_cam1 = process_query_sysu(data_dir, mode='all')
        gall_img1, gall_label1, gall_cam1 = process_gallery_sysu(data_dir, mode='all', trial=0)
        
        gallset_all  = TestData(gall_img1, gall_label1, img_h=args.img_h, img_w=args.img_w)
        queryset_all = TestData(query_img1, query_label1, img_h=args.img_h, img_w=args.img_w)
        
        query_img2, query_label2, query_cam2 = process_query_sysu(data_dir, mode='indoor')
        gall_img2, gall_label2, gall_cam2 = process_gallery_sysu(data_dir, mode='indoor', trial=0)
        
        gallset_indoor  = TestData(gall_img2, gall_label2, img_h=args.img_h, img_w=args.img_w)
        queryset_indoor = TestData(query_img2, query_label2, img_h=args.img_h, img_w=args.img_w)
        
        gall_loader_all = data.DataLoader(gallset_all, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
        query_loader_all = data.DataLoader(queryset_all, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
        nquery_all = len(query_label1)
        ngall_all = len(gall_label1)
        
        gall_loader_indoor = data.DataLoader(gallset_indoor, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
        query_loader_indoor = data.DataLoader(queryset_indoor, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
        nquery_indoor = len(query_label2)
        ngall_indoor = len(gall_label2)
    
        print("  ----------------------------")
        print("  ALL SEARCH ")
        print("  query    | {:5d} | {:8d}".format(len(np.unique(query_label1)), len(query_label1)))
        print("  gallery  | {:5d} | {:8d}".format(len(np.unique(gall_label1)), len(gall_label1)))
        print("  INDOOR SEARCH ")
        print("  query    | {:5d} | {:8d}".format(len(np.unique(query_label2)), len(query_label2)))
        print("  gallery  | {:5d} | {:8d}".format(len(np.unique(gall_label2)), len(gall_label2)))
        print("  ----------------------------")
        
    elif args.dataset == 'regdb':
        
        ## parameters for regdb dataset
        args.eps = 0.3
        args.train_iter = 100
                
        data_dir = osp.join(args.data_dir, 'RegDB')
        dataset = RegDB(args, args.data_dir) 
        
        query_img1, query_label1 = process_test_regdb(data_dir, trial=args.regdb_trial, modal='visible') ## query : visible 
        gall_img1, gall_label1 = process_test_regdb(data_dir, trial=args.regdb_trial, modal='thermal') ## gallery : thermal
        
        query_img2, query_label2 = process_test_regdb(data_dir, trial=args.regdb_trial, modal='thermal') ## query : visible 
        gall_img2, gall_label2 = process_test_regdb(data_dir, trial=args.regdb_trial, modal='visible') ## gallery : thermal
        
        gallset_v2t = TestData(gall_img1, gall_label1, img_h=args.img_h, img_w=args.img_w)
        queryset_v2t = TestData(query_img1, query_label1, img_h=args.img_h, img_w=args.img_w)
        
        gallset_t2v  = TestData(gall_img2, gall_label2, img_h=args.img_h, img_w=args.img_w)
        queryset_t2v = TestData(query_img2, query_label2, img_h=args.img_h, img_w=args.img_w)
        
        print("  ----------------------------")
        print("   visibletothermal   ")
        print("  query    | {:5d} | {:8d}".format(len(np.unique(query_label1)), len(query_label1)))
        print("  gallery  | {:5d} | {:8d}".format(len(np.unique(gall_label1)), len(gall_label1)))
        print("   thermaltovisible   ")
        print("  query    | {:5d} | {:8d}".format(len(np.unique(query_label2)), len(query_label2)))
        print("  gallery  | {:5d} | {:8d}".format(len(np.unique(gall_label2)), len(gall_label2)))
        print("  ----------------------------")
    
        # testing data loader
        gall_loader_v2t = data.DataLoader(gallset_v2t, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
        query_loader_v2t = data.DataLoader(queryset_v2t, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
        nquery_v2t = len(query_label1)
        ngall_v2t = len(gall_label1)
        
        gall_loader_t2v = data.DataLoader(gallset_t2v, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
        query_loader_t2v = data.DataLoader(queryset_t2v, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
        nquery_t2v = len(query_label2)
        ngall_t2v = len(gall_label2)
    
    ## model
    print(args.device)
    device = torch.device( f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    device, torch.cuda.device_count(), torch.cuda.is_available(), torch.cuda.current_device(), torch.cuda.get_device_properties(1)

    print('==> Building model..')
    main_net = BaseResNet(args, class_num=0, non_local='off', gm_pool='on', per_add_iters=args.per_add_iters)
    main_net.to(device)
    if args.pretrained:
        device_ids=[args.device]
    else:
        device_ids=[args.device, args.device + 1]
    main_net = nn.DataParallel(main_net, device_ids=device_ids)

    print('The network is pretrained under single modality: {}'.format(args.pretrained))
    if args.pretrained == True:
        if args.dataset == 'sysu':
            checkpoint = load_checkpoint(osp.join(osp.dirname(osp.abspath(__file__)), 
                        'logs/base2/stage_two_model', 'checkpoint.pth.tar'))
        elif args.dataset == 'regdb':
            checkpoint = load_checkpoint(osp.join(osp.dirname(osp.abspath(__file__)), 
                        'logs/regdb/trial' + str(args.regdb_trial) +'_base/stage_two_model', 'checkpoint.pth.tar'))
        copy_state_dict(checkpoint['state_dict'], main_net.module, strip='module.')


    main_net.module.attn_pool = SAFL(dim=2048, part_num=args.part_num).to(device)
    main_net.module.bottleneck_part = nn.BatchNorm1d(2048 * args.part_num).to(device) 
    main_net.module.bottleneck_part.bias.requires_grad_(False)  # no shift
    main_net.module.bottleneck_part.apply(weights_init_kaiming)

        
    ## build optimizer and trainer
    optimizer = select_optimizer(args, main_net)

    trainer = AssignTrainer(args=args, encoder=main_net, batch_size=args.batch_size, num_pos=args.num_pos)
    trainer.temp = args.temp
    
    assign_RGB_set = dataset_for_feature_extractor(dataset.train_rgb, img_h=args.img_h, img_w=args.img_w)
    assign_IR_set = dataset_for_feature_extractor(dataset.train_ir, img_h=args.img_h, img_w=args.img_w)
    assign_RGB_loader = data.DataLoader(assign_RGB_set, batch_size=args.test_batch, num_workers=args.workers, drop_last=False)
    assign_IR_loader = data.DataLoader(assign_IR_set, batch_size=args.test_batch, num_workers=args.workers, drop_last=False)

    best_mAP = 0
    ## training
    
    if args.pretrained == True:
        num_epochs = args.epochs - args.stage_one
    else:
        num_epochs = args.stage_one

    ari_v, ari_r = [], []
    fmi_v, fmi_r = [], []
    ami_v, ami_r = [], []
    v_measure_v, v_measure_r = [], []

    for epoch in range(num_epochs):
        
        if args.pretrained == True:
            epoch = epoch + args.stage_one
            if epoch == args.stage_one:
                eps = args.eps
                print('Clustering criterion: eps: {:.3f}'.format(eps))
                cluster = DBSCAN(eps=eps, min_samples=4, metric='precomputed', n_jobs=-1)
            
        if epoch == 0:
            # DBSCAN cluster
            eps = args.eps
            print('Clustering criterion: eps: {:.3f}'.format(eps))
            cluster = DBSCAN(eps=eps, min_samples=4, metric='precomputed', n_jobs=-1)

        ## extracting features and clustering
        print('==> start feature extracting...')
        features_RGB, features_RGB_pg, gt_labels_RGB = extract_features_for_cluster(main_net, assign_RGB_loader, mode='RGB', device=args.device)
        features_IR, features_IR_pg, gt_labels_IR = extract_features_for_cluster(main_net, assign_IR_loader, mode='IR', device=args.device)

        features_RGB = torch.cat([features_RGB[f].unsqueeze(0) for f, _, _ in sorted(dataset.train_rgb)], 0)
        features_IR = torch.cat([features_IR[f].unsqueeze(0) for f, _, _ in sorted(dataset.train_ir)], 0)

        features_RGB_pg = torch.cat([features_RGB_pg[f].unsqueeze(0) for f, _, _ in sorted(dataset.train_rgb)], 0)
        features_IR_pg = torch.cat([features_IR_pg[f].unsqueeze(0) for f, _, _ in sorted(dataset.train_ir)], 0)

        dim = features_RGB.shape[1]
        
        gt_labels_RGB = torch.cat([torch.tensor(gt_labels_RGB[f]).unsqueeze(0) for f, _, _ in sorted(dataset.train_rgb)])
        gt_labels_IR = torch.cat([torch.tensor(gt_labels_IR[f]).unsqueeze(0) for f, _, _ in sorted(dataset.train_ir)])

        labels_RGB = dbscan_cluster(args, features_RGB, cluster=cluster)  
        labels_IR = dbscan_cluster(args, features_IR, cluster=cluster)  

        pseudo_dataset_RGB, pseudo_dataset_IR = [], []
        centers_RGB = generate_center(labels_RGB, features_RGB)
        centers_IR = generate_center(labels_IR, features_IR)

        centers_RGB_pg = generate_center(labels_RGB, features_RGB_pg)
        centers_IR_pg = generate_center(labels_IR, features_IR_pg)

        mask_v = torch.as_tensor((labels_RGB != -1), dtype=torch.float)
        mask_r = torch.as_tensor((labels_IR != -1), dtype=torch.float)

        for i, ((fname, _, _), label, mask) in enumerate(zip(sorted(dataset.train_rgb), labels_RGB, mask_v)): 
            if mask == 1:
                pseudo_dataset_RGB.append((fname, label))

        for i, ((fname, _, _), label, mask) in enumerate(zip(sorted(dataset.train_ir), labels_IR, mask_r)): 
            if mask == 1:
                pseudo_dataset_IR.append((fname, label))

        labels_RGB = labels_RGB[mask_v==1]
        labels_IR = labels_IR[mask_r==1]
        features_RGB, features_IR = features_RGB[mask_v==1], features_IR[mask_r==1]
        features_RGB_pg, features_IR_pg = features_RGB_pg[mask_v==1], features_IR_pg[mask_r==1]
        gt_labels_RGB, gt_labels_IR = gt_labels_RGB[mask_v==1], gt_labels_IR[mask_r==1]
        N_v, N_r = features_RGB.shape[0], features_IR.shape[0]

        ## assignment and build new dataset
        if epoch < args.stage_one:
            test_part = False
            print_intra_acc(gt_labels_RGB, gt_labels_IR, labels_RGB, labels_IR)
            num_cls_rgb = centers_RGB.shape[0]
            num_cls_ir = centers_IR.shape[0]

            dataset_train = pseudo_label_dataset(args, pseudo_dataset_RGB, pseudo_dataset_IR,
                                                img_h=args.img_h, img_w=args.img_w, epoch=epoch, stage='single') 
            dataset_train.label_RGB = np.array(labels_RGB)
            dataset_train.label_IR = np.array(labels_IR)     

        else:

            test_part = False
            labels_RGB, labels_IR, cm_labels_RGB, cm_labels_IR, y_rgb, y_ir, y_rgb_cm, y_ir_cm, \
            y_rgb_p, y_ir_p, y_rgb_cm_p, y_ir_cm_p, _\
            = part_ot(args, epoch, features_RGB_pg, features_IR_pg, labels_RGB, labels_IR, centers_RGB_pg, centers_IR_pg)
            
            print('-------print accuracy of cross-modality label------')
            print_intra_acc(gt_labels_RGB, gt_labels_IR, labels_RGB, labels_IR)
            print_intra_acc(gt_labels_RGB, gt_labels_IR, cm_labels_RGB, cm_labels_IR)
            print_cm_acc(gt_labels_RGB, gt_labels_IR, labels_RGB, labels_IR, cm_labels_RGB, cm_labels_IR)

            ari_v_epoch = metrics.adjusted_rand_score(torch.cat((gt_labels_RGB,gt_labels_IR),dim=0).numpy(), 
                                            torch.cat((labels_RGB, cm_labels_IR),dim=0).numpy())
            ari_r_epoch = metrics.adjusted_rand_score(torch.cat((gt_labels_RGB,gt_labels_IR),dim=0).numpy(), 
                                            torch.cat((cm_labels_RGB, labels_IR),dim=0).numpy())
            ari_v.append(ari_v_epoch)
            ari_r.append(ari_r_epoch)
            print('ARI-score:{:.4f}, {:.4f}'.format(ari_v_epoch, ari_r_epoch))

            fmi_v_epoch = metrics.fowlkes_mallows_score(torch.cat((gt_labels_RGB,gt_labels_IR),dim=0).numpy(), 
                                            torch.cat((labels_RGB, cm_labels_IR),dim=0).numpy())
            fmi_r_epoch = metrics.fowlkes_mallows_score(torch.cat((gt_labels_RGB,gt_labels_IR),dim=0).numpy(), 
                                            torch.cat((cm_labels_RGB, labels_IR),dim=0).numpy())
            fmi_v.append(fmi_v_epoch)
            fmi_r.append(fmi_r_epoch)
            print('FMI-score:{:.4f}, {:.4f}'.format(fmi_v_epoch, fmi_r_epoch))

            ami_v_epoch = metrics.adjusted_mutual_info_score(torch.cat((gt_labels_RGB,gt_labels_IR),dim=0).numpy(), 
                                            torch.cat((labels_RGB, cm_labels_IR),dim=0).numpy())
            ami_r_epoch = metrics.adjusted_mutual_info_score(torch.cat((gt_labels_RGB,gt_labels_IR),dim=0).numpy(), 
                                            torch.cat((cm_labels_RGB, labels_IR),dim=0).numpy())
            ami_v.append(ami_v_epoch)
            ami_r.append(ami_r_epoch)
            print('AMI-score:{:.4f}, {:.4f}'.format(ami_v_epoch, ami_r_epoch))

            v_measure_v_epoch = metrics.v_measure_score(torch.cat((gt_labels_RGB,gt_labels_IR),dim=0).numpy(), 
                                            torch.cat((labels_RGB, cm_labels_IR),dim=0).numpy())
            v_measure_r_epoch = metrics.v_measure_score(torch.cat((gt_labels_RGB,gt_labels_IR),dim=0).numpy(), 
                                            torch.cat((cm_labels_RGB, labels_IR),dim=0).numpy())
            v_measure_v.append(v_measure_v_epoch)
            v_measure_r.append(v_measure_r_epoch)
            print('V-measure-score:{:.4f}, {:.4f}'.format(v_measure_v_epoch, v_measure_r_epoch))
            
            dataset_train = pseudo_label_dataset(args, pseudo_dataset_RGB, pseudo_dataset_IR, 
                                    cm_labels_RGB, cm_labels_IR,
                                    img_h=args.img_h, img_w=args.img_w, epoch=epoch, stage='cross') 
            dataset_train.label_RGB = labels_RGB
            dataset_train.label_IR = labels_IR   

            dataset_train.y_rgb = y_rgb
            dataset_train.y_ir = y_ir
            dataset_train.y_rgb_cm = y_rgb_cm
            dataset_train.y_ir_cm = y_ir_cm

            dataset_train.y_rgb_p = y_rgb_p
            dataset_train.y_ir_p = y_ir_p
            dataset_train.y_rgb_cm_p = y_rgb_cm_p
            dataset_train.y_ir_cm_p = y_ir_cm_p
            dataset_train.N_v = N_v

        print('Training stage: {}'.format(dataset_train.stage))
        
        
        ##build memory
        m_v = ClusterMemory(centers_RGB.shape[1], centers_RGB.shape[0], temp=args.temp,
                                    momentum=args.momentum, use_hard=args.use_hard).to(device)
        m_r = ClusterMemory(centers_IR.shape[1], centers_IR.shape[0], temp=args.temp,
                                    momentum=args.momentum, use_hard=args.use_hard).to(device)

        if epoch >= args.stage_one:

            m_vi = ClusterMemory(centers_RGB.shape[1], N_v, temp=args.temp,
                                        momentum=args.momentum, use_hard=args.use_hard).to(device)
            m_ri = ClusterMemory(centers_IR.shape[1], N_r, temp=args.temp,
                                        momentum=args.momentum, use_hard=args.use_hard).to(device)
            
            m_vp = [ClusterMemory(centers_RGB.shape[1], centers_RGB.shape[0], temp=args.temp,
                                momentum=args.momentum_cross, use_hard=args.use_hard).to(device) for i in range(args.part_num)]
            m_rp = [ClusterMemory(centers_RGB.shape[1], centers_RGB.shape[0], temp=args.temp,
                                momentum=args.momentum_cross, use_hard=args.use_hard).to(device) for i in range(args.part_num)]
            m_vpa = [ClusterMemory(centers_RGB.shape[1], centers_RGB.shape[0], temp=args.temp,
                                momentum=args.momentum_cross, use_hard=args.use_hard).to(device) for i in range(args.part_num)]
            m_rpa = [ClusterMemory(centers_RGB.shape[1], centers_RGB.shape[0], temp=args.temp,
                                momentum=args.momentum_cross, use_hard=args.use_hard).to(device) for i in range(args.part_num)]
            
            m_vpi = [ClusterMemory(centers_RGB.shape[1], N_v, temp=args.temp,
                                momentum=args.momentum_cross, use_hard=args.use_hard).to(device) for i in range(args.part_num)]
            m_rpi = [ClusterMemory(centers_RGB.shape[1], N_r, temp=args.temp,
                                momentum=args.momentum_cross, use_hard=args.use_hard).to(device) for i in range(args.part_num)]

        print(centers_RGB.shape)
        m_v.features = F.normalize(centers_RGB, dim=1).to(device)
        m_r.features = F.normalize(centers_IR, dim=1).to(device)

        if epoch >= args.stage_one:
            m_vi.features = F.normalize(features_RGB, dim=1).to(device)
            m_ri.features = F.normalize(features_IR, dim=1).to(device)

            for i in range(args.part_num):
                m_vp[i].features = F.normalize(centers_RGB_pg[:, i*dim : (i+1)*dim], dim=1).to(device)
                m_rp[i].features = F.normalize(centers_IR_pg[:, i*dim : (i+1)*dim], dim=1).to(device)
                m_vpa[i].features = F.normalize(centers_RGB_pg[:, i*dim : (i+1)*dim], dim=1).to(device)
                m_rpa[i].features = F.normalize(centers_IR_pg[:, i*dim : (i+1)*dim], dim=1).to(device)

                m_vpi[i].features = F.normalize(features_RGB_pg[:, i*dim : (i+1)*dim], dim=1).to(device)
                m_rpi[i].features = F.normalize(features_IR_pg[:, i*dim : (i+1)*dim], dim=1).to(device)

        trainer.m_v = m_v
        trainer.m_r = m_r

        if epoch >= args.stage_one:
            trainer.m_vi = m_vi
            trainer.m_ri = m_ri

            trainer.m_vp = m_vp
            trainer.m_rp = m_rp
            trainer.m_vpa = m_vpa
            trainer.m_rpa = m_rpa

            trainer.m_vpi = m_vpi
            trainer.m_rpi = m_rpi
        
        ## build train loader
        stage1_num_pos, stage1_batch_size = 12, 12
        if dataset_train.stage == 'single':

            color_pos, thermal_pos = GenIdx(labels_RGB, labels_IR)
            sampler = StageOneSampler(args, dataset_train.label_RGB, dataset_train.label_IR, 
                                    color_pos, thermal_pos, num_pos=stage1_num_pos, batchSize=stage1_batch_size)

        elif dataset_train.stage == 'cross':

            color_pos, thermal_pos = GenIdx(labels_RGB, labels_IR)
            cross_color_pos, cross_thermal_pos = GenIdx(cm_labels_RGB, cm_labels_IR)
            
            sampler = StageThreeSampler(
                args, dataset_train.label_RGB, dataset_train.label_IR, cm_labels_RGB, cm_labels_IR,
                color_pos, thermal_pos, cross_color_pos, cross_thermal_pos, args.num_pos, args.batch_size
            )
                
        dataset_train.cIndex = sampler.index1  # color index
        dataset_train.tIndex = sampler.index2  # thermal index
        if dataset_train.stage == 'single':
            trainloader = data.DataLoader(dataset_train, batch_size=stage1_num_pos*stage1_batch_size,
                                        sampler=sampler, num_workers=args.workers, drop_last=True)
        if dataset_train.stage == 'cross':
            trainloader = data.DataLoader(dataset_train, batch_size=args.batch_size * args.num_pos, 
                                        sampler=sampler, num_workers=args.workers, drop_last=True)
        
        ## train
        print('==> start training...')
        trainer.train(args, epoch, trainloader, optimizer, device, stage=dataset_train.stage)

        ## evaluate
        if (epoch + 1) % args.eval_step == 0:
            print('Test Epoch: {}'.format(epoch))
            if args.dataset == 'sysu':
                cmc_all, mAP_all, mINP_all = test(args, main_net,  
                                    ngall_all, nquery_all, gall_loader_all, query_loader_all, 
                                    query_label1, gall_label1, query_cam=query_cam1, gall_cam=gall_cam1, part=test_part)
                
                cmc_indoor, mAP_indoor, mINP_indoor = test(args, main_net,  
                                    ngall_indoor, nquery_indoor, gall_loader_indoor, query_loader_indoor, 
                                    query_label2, gall_label2, query_cam=query_cam2, gall_cam=gall_cam2, part=test_part)
                mAP = mAP_all
                
                print('all:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                    cmc_all[0], cmc_all[4], cmc_all[9], cmc_all[19], mAP_all, mINP_all))
                
                print('indoor:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                    cmc_indoor[0], cmc_indoor[4], cmc_indoor[9], cmc_indoor[19], mAP_indoor, mINP_indoor))
                                
            elif args.dataset == 'regdb':
                cmc_v2t, mAP_v2t, mINP_v2t = test(args, main_net, 
                                      ngall_v2t, nquery_v2t,
                                      gall_loader_v2t, query_loader_v2t, 
                                      query_label1, gall_label1, test_mode=['IR', 'RGB'], part=test_part)
                
                cmc_t2v, mAP_t2v, mINP_t2v = test(args, main_net, 
                                      ngall_t2v, nquery_t2v,
                                      gall_loader_t2v, query_loader_t2v, 
                                      query_label2, gall_label2, test_mode=['RGB', 'IR'], part=test_part)
                mAP = mAP_v2t
                
                print('VisibleToThermal:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                    cmc_v2t[0], cmc_v2t[4], cmc_v2t[9], cmc_v2t[19], mAP_v2t, mINP_v2t))
                
                print('ThermalToVisible:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                    cmc_t2v[0], cmc_t2v[4], cmc_t2v[9], cmc_t2v[19], mAP_t2v, mINP_t2v))
            
        if epoch + 1 ==  args.stage_one:
            print('Save model for stage one')
            
            save_checkpoint({
                'state_dict': main_net.module.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
            }, is_best=True, fpath=osp.join(args.logs_dir, 'stage_one_model', 'checkpoint.pth.tar'))
            
        if (epoch + 1) % args.eval_step == 0 or (epoch + 1) ==  args.epochs:

            print('Save final model')

            save_checkpoint({
                'state_dict': main_net.module.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
            }, is_best=True, fpath=osp.join(args.logs_dir, 'stage_two_model', 'checkpoint.pth.tar'))

        if args.dataset == 'sysu':
            is_best = (mAP > best_mAP)
            best_mAP = max(mAP, best_mAP)
            # print(is_best)
            save_checkpoint({
                'state_dict': main_net.module.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
            }, is_best, fpath=osp.join(args.logs_dir, 'best_model', 'checkpoint.pth.tar'))
            print('Best mAP for trial 0: {:.2%}'.format(best_mAP))

    np.save('hlf_ReID/USL/cluster_metric/base/ari_v.npy', np.array(ari_v))
    np.save('hlf_ReID/USL/cluster_metric/base/ari_r.npy', np.array(ari_r))
    np.save('hlf_ReID/USL/cluster_metric/base/fmi_v.npy', np.array(fmi_v))
    np.save('hlf_ReID/USL/cluster_metric/base/fmi_r.npy', np.array(fmi_r))
    np.save('hlf_ReID/USL/cluster_metric/base/ami_v.npy', np.array(ami_v))
    np.save('hlf_ReID/USL/cluster_metric/base/ami_r.npy', np.array(ami_r))
    np.save('hlf_ReID/USL/cluster_metric/base/v_measure_v.npy', np.array(v_measure_v))
    np.save('hlf_ReID/USL/cluster_metric/base/v_measure_r.npy', np.array(v_measure_r))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="assignment main train")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--regdb-trial', type=int, default=5)

    parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
    
    parser.add_argument('--dataset', default='sysu', help='dataset name: regdb or sysu]') ### DATASET
    parser.add_argument('--mode', default='all', type=str, help='sysu:all/indoor regdb:visibletothermal')

    # sysu: all / indoor
    # regdb: visible-to-thermal / thermal-to-visible
    parser.add_argument('--epochs', default=80)
    parser.add_argument('--img_w', default=144, type=int,
                    metavar='imgw', help='img width')
    parser.add_argument('--img_h', default=288, type=int,
                    metavar='imgh', help='img height')
    parser.add_argument('--test-batch', default=256, type=int,
                    metavar='tb', help='testing batch size')
    parser.add_argument('--batch-size', default=8, type=int,
                    metavar='B', help='training batch size')
    parser.add_argument('--num_pos', default=8, type=int, 
                    help='num of pos per identity in each modality')
    parser.add_argument('--print-step', default=50, type=int)
    parser.add_argument('--eval-step', default=1, type=int) ## 1 for SYSU-MM01 / 10 for RegDB
    parser.add_argument('--stage-one', default=40, type=int)
    
    ## cluster
    parser.add_argument('--eps', type=float, default=0.6,
                        help="max neighbor distance for DBSCAN") ## 0.6 for sysu and 0.3 for regdb
    parser.add_argument('--k1', type=int, default=30,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--k2', type=int, default=6,
                        help="hyperparameter for jaccard distance")
    
    ## network  
    parser.add_argument('--arch', default='resnet50')
    parser.add_argument('--pool-dim', default=2048)
    parser.add_argument('--per-add-iters', default=1, help='param for GRL')
    parser.add_argument('--lr', default=0.00035, help='learning rate for main net')
    parser.add_argument('--optim', default='adam', help='optimizer')
    parser.add_argument('--pooling-type', type=str, default='gem')
    parser.add_argument('--train-iter', type=int, default=300) ## 100 for regdb and 300 for sysu
    parser.add_argument('--pretrained', type=bool, default=True)
    
    ## memory
    parser.add_argument('--momentum', type=float, default=0.1,
                        help="update momentum for the hybrid memory")
    parser.add_argument('--momentum-cross', type=float, default=0.1,
                        help="update momentum for the hybrid memory")
    parser.add_argument('--temp', type=float, default=0.05,
                        help="temperature for scaling contrastive loss")
    parser.add_argument('--use-hard', default=False)

    ## hyper-parameters for our framework
    parser.add_argument('--lambda-ot', default=5, help = 'hyper-petermeter for optimal transport') ## fixed
    parser.add_argument('--k-cmff', type=int, default=30, help = 'KNN neighbor for feature augmentation before OTLA') 
    parser.add_argument('--k-cmff-test', type=int, default=30, help = 'KNN neighbor for CMFP during test stage') 
    parser.add_argument('--part_num', type=int, default=3, help = 'number of part prototypes in baseline network') 
    parser.add_argument('--lambda-1', type=float, default=1.0, help= 'weight for part contrastive loss') 
    parser.add_argument('--lambda-2', type=float, default=0.5, help= 'weight for instance-memory online refinement loss') 
    parser.add_argument('--k-instance', type=float, default=30, help= 'number of neighbors for instance memory bank') 

    ## sysu: k_cmff=30; k_cmff_test=30; part_num=3; lambda_2=0.5; k_instance=30 //
    ## regdb: k_cmff=8; k_cmff_test=8; part_num=3; lambda_2=0.5; k_instance=30 //

    ## components of method for ablation study
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--branch', type=str, default='dual', help = 'perform cross-modality feature fusion')
    parser.add_argument('--CMFP', type=bool, default=True, help = 'perform cross-modality feature fusion')
    parser.add_argument('--FGSAL', type=bool, default=True, help = 'utilize part prototype strategy')
    parser.add_argument('--GPCR', type=bool, default=True, help = 'utilize part prototype strategy')

    ## utilize re-ranking technology during test
    parser.add_argument('--test_AIM', type=bool, default=False, help="Use CMRR in evaluation")
    parser.add_argument('--test_CMFP', type=bool, default=True, help="Use CMRR in evaluation")
    parser.add_argument('--test_CMRR', type=bool, default=False, help="Use CMRR in evaluation")

    ## path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default='')
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs/sysu/SALCR_model'))
    main()


    

        
    
