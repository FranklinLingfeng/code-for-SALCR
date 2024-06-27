'''
Author: error: git config user.name && git config user.email & please set dead value or install git
Date: 2023-01-17 00:35:06
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2023-02-04 10:46:19
FilePath: /Lingfeng He/xiongyali_new_idea/assignment/evaluator.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import time
from collections import OrderedDict
import numpy as np
import torch
from utils.meters import AverageMeter
from torch.autograd import Variable
from cross_modality_rerank import re_ranking_cross
import torch.nn.functional as F


def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip


def pairwise_distance(x, y):
   
    x, y = torch.tensor(x), torch.tensor(y)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist_mat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(x, y.t(), beta=1, alpha=-2)
        
    return dist_mat.numpy()


def cal_distmat(x, y):
    # compute dismat
    x = torch.tensor(x)
    y = torch.tensor(y)
    m, n = x.shape[0], y.shape[0]
    x = x.view(m, -1)
    y = y.view(n, -1)

    distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, x, y.t())
    return distmat.numpy()



def getNewFeature(x, y, k1, k2, mean: bool = False):
    dismat = x @ y.T
    val, rank = dismat.topk(k1)
    dismat[dismat < val[:, -1].unsqueeze(1)] = 0
    if mean:
        dismat = dismat[rank[:, :k2]].mean(dim=1)
    return dismat


def AIM(qf: torch.tensor, gf: torch.tensor, k1, k2):
    qf = qf.to('cuda')
    gf = gf.to('cuda')

    qf = torch.nn.functional.normalize(qf)
    gf = torch.nn.functional.normalize(gf)

    new_qf = torch.concat([getNewFeature(qf, gf, k1, k2)], dim=1)
    new_gf = torch.concat([getNewFeature(gf, gf, k1, k2, mean=True)], dim=1)

    new_qf = torch.nn.functional.normalize(new_qf)
    new_gf = torch.nn.functional.normalize(new_gf)

    return (-new_qf @ new_gf.T - qf @ gf.T).to('cpu')




def extract_cnn_feature(model, inputs, mode, device):
    inputs = inputs.to(device)
    outputs_g, outputs_pg = model(inputs, inputs, mode=mode) ## 256 * 2048
    outputs_g = outputs_g.data.cpu()
    outputs_pg = outputs_pg.data.cpu()
    
    return outputs_g, outputs_pg


def knn_graph(init_graph, k=30):

    threshold = init_graph.sort(dim=1, descending=True)[0][:, k-1]
    mask = torch.as_tensor(init_graph >= threshold.unsqueeze(1), dtype = torch.float)
    graph = mask * init_graph
    graph = graph.clamp(min=0)
    graph = graph / graph.sum(1).unsqueeze(1)

    return graph


def agg_feature(features_RGB, features_IR, k=30):
    
    tt = time.time()
    print('Start knn-search:')
    A_vv = knn_graph(features_RGB.mm(features_RGB.t()), k=k)
    A_vr = knn_graph(features_RGB.mm(features_IR.t()), k=k)
    A_rv = knn_graph(features_IR.mm(features_RGB.t()), k=k)
    A_rr = knn_graph(features_IR.mm(features_IR.t()), k=k)
    print('Finish knn-search:{}'.format(time.time()-tt))

    A = torch.cat((torch.cat((A_vv, A_vr), dim=1), torch.cat((A_rv, A_rr), dim=1)), dim=0)
    features_ = A.mm(torch.cat((features_RGB, features_IR), dim=0))
    features_ = F.normalize(features_, dim=1)

    return A, features_


def extract_features_for_cluster(model, data_loader, print_freq=20, mode=None, device=0):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features_g = OrderedDict() 
    features_pg = OrderedDict() 
    labels = OrderedDict()

    end = time.time()
    with torch.no_grad():
        for i, (fnames, imgs, pids) in enumerate(data_loader):
            data_time.update(time.time() - end)
            outputs_g, outputs_pg = extract_cnn_feature(model, imgs, mode, device)

            for fname, output_g, output_pg, pid in zip(fnames, outputs_g, outputs_pg, pids):
                features_g[fname] = output_g
                features_pg[fname] = output_pg
                labels[fname] = int(pid)

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Extract Features: [{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      .format(i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg))

    return features_g, features_pg, labels


def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip


def test(args, model, ngall, nquery, gall_loader, query_loader, 
         query_label, gall_label, query_cam=None, gall_cam=None, test_mode=None, part=False):
    # switch to evaluation mode
    if args.dataset == 'sysu':
        test_mode = ['RGB', 'IR']
    elif args.dataset == 'regdb' or args.dataset == 'llcm':
        test_mode = test_mode
    
    dim=2048
    if part == True:
        dim = 2048 * (args.part_num+1)
    model.eval()
    
    print('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    gall_feat = np.zeros((ngall, dim))
    gall_pool = np.zeros((ngall, dim))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(gall_loader):
            batch_num = input.size(0)
            flip_input = fliplr(input)
            input = Variable(input.to(args.device))
            feat, feat_pg = extract_cnn_feature(model, input, test_mode[0], args.device)
            flip_input = Variable(flip_input.to(args.device))
            flip_feat, flip_feat_pg = extract_cnn_feature(model,flip_input, test_mode[0], args.device)
            if part == True:
                feat, flip_feat = feat_pg, flip_feat_pg
            feature_fc = (feat.detach() + flip_feat.detach())/2
            fnorm_fc = torch.norm(feature_fc, p=2, dim=1, keepdim=True)
            feature_fc = feature_fc.div(fnorm_fc.expand_as(feature_fc))
            gall_feat[ptr:ptr + batch_num, :] = feature_fc.cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    # switch to evaluation
    model.eval()


    print('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    query_feat = np.zeros((nquery, dim))
    query_pool = np.zeros((nquery, dim))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(query_loader):
            batch_num = input.size(0)
            flip_input = fliplr(input)
            input = Variable(input.to(args.device))
            feat, feat_pg = extract_cnn_feature(model, input, test_mode[1], args.device)
            flip_input = Variable(flip_input.to(args.device))
            flip_feat, flip_feat_pg = extract_cnn_feature(model,flip_input, test_mode[1], args.device)
            if part == True:
                feat, flip_feat = feat_pg, flip_feat_pg
            feature_fc = (feat.detach() + flip_feat.detach())/2
            fnorm_fc = torch.norm(feature_fc, p=2, dim=1, keepdim=True)
            feature_fc = feature_fc.div(fnorm_fc.expand_as(feature_fc))
            query_feat[ptr:ptr + batch_num, :] = feature_fc.cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    start = time.time()
    # compute the similarity

    if args.test_CMFF == True:
        tt = time.time()
        print("Start cross-modality feature fusion for evaluation!")
        A, features = agg_feature(torch.tensor(query_feat), torch.tensor(gall_feat), k=args.k_cmff_test)
        query_feat = F.normalize(features[:nquery], dim=1).numpy()
        gall_feat = F.normalize(features[nquery:], dim=1).numpy()
        print("Finiah cross-modality feature fusion, time:{}".format(time.time()-tt))
    
    distmat = pairwise_distance(query_feat, gall_feat)

    if args.test_CMRR == True:
        tt = time.time()
        print("Start CMRR-Reranking for evaluation!")
        hetero_features_ir = query_feat
        hetero_features_rgb = gall_feat
        homo_features_ir = query_feat
        homo_features_rgb = gall_feat

        distmat_ir_rgb = cal_distmat(hetero_features_ir, hetero_features_rgb)
        distmat_rgb_rgb = cal_distmat(homo_features_rgb, homo_features_rgb)
        distmat_ir_ir = cal_distmat(homo_features_ir, homo_features_ir)

        if args.dataset == 'sysu':
            k_for_cmrr = 15
        elif args.dataset == 'regdb':
            k_for_cmrr = 8

        rerank_dist, _ = re_ranking_cross(distmat_ir_rgb, distmat_ir_rgb.T, 
                                        distmat_ir_ir, distmat_rgb_rgb,
                                        hetero_features_ir, hetero_features_rgb, 
                                        k=k_for_cmrr,eta_value=0.1)
        distmat = rerank_dist
        print("Finiah CMRR-Reranking, time:{}".format(time.time()-tt))


    if args.test_AIM == True:
        tt = time.time()
        print("Start AIM for evaluation!")
        if args.dataset == 'sysu':
            distmat = AIM(qf=torch.tensor(query_feat), gf=torch.tensor(gall_feat), k1=4, k2=1)
        elif args.dataset == 'regdb':
            distmat = AIM(qf=torch.tensor(query_feat), gf=torch.tensor(gall_feat), k1=8, k2=2)
        
        print("Finiah AIM, time:{}".format(time.time()-tt))
    
    # evaluation
    print('eval feat after batchnorm')

    if args.dataset == 'sysu':
        cmc, mAP, mINP = eval_sysu(distmat, query_label, gall_label, query_cam, gall_cam)
    elif args.dataset == 'regdb':
        cmc, mAP, mINP = eval_regdb(distmat, query_label, gall_label)
    elif args.dataset == 'llcm':
        cmc, mAP, mINP = eval_llcm(distmat, query_label, gall_label, query_cam, gall_cam)
    print('Evaluation Time:\t {:.3f}'.format(time.time() - start))
    return cmc, mAP, mINP





def eval_sysu(distmat, q_pids, g_pids, q_camids, g_camids, max_rank = 20):
    """Evaluation with sysu metric
    Key: for each query identity, its gallery images from the same camera view are discarded. "Following the original setting in ite dataset"
    """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    pred_label = g_pids[indices]
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    
    # compute cmc curve for each query
    new_all_cmc = []
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0. # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (q_camid == 3) & (g_camids[order] == 2)
        keep = np.invert(remove)
        
        # compute cmc curve
        # the cmc calculation is different from standard protocol
        # we follow the protocol of the author's released code
        new_cmc = pred_label[q_idx][keep]
        new_index = np.unique(new_cmc, return_index=True)[1]
        new_cmc = [new_cmc[index] for index in sorted(new_index)]
        
        new_match = (new_cmc == q_pid).astype(np.int32)
        new_cmc = new_match.cumsum()
        new_all_cmc.append(new_cmc[:max_rank])
        
        orig_cmc = matches[q_idx][keep] # binary vector, positions with value 1 are correct matches
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()

        # compute mINP
        # refernece Deep Learning for Person Re-identification: A Survey and Outlook
        pos_idx = np.where(orig_cmc == 1)
        pos_max_idx = np.max(pos_idx)
        inp = cmc[pos_max_idx]/ (pos_max_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"
    
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q   # standard CMC
    
    new_all_cmc = np.asarray(new_all_cmc).astype(np.float32)
    new_all_cmc = new_all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)
    return new_all_cmc, mAP, mINP


def eval_regdb(distmat, q_pids, g_pids, max_rank=20):
    """
    Evaluation with RegDB metric.
    """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0.  # number of valid query

    # only two cameras
    q_camids = np.ones(num_q).astype(np.int32)
    g_camids = 2 * np.ones(num_g).astype(np.int32)

    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        raw_cmc = matches[q_idx][keep]  # binary vector, positions with value 1 are correct matches
        if not np.any(raw_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = raw_cmc.cumsum()

        # compute mINP
        # refernece Deep Learning for Person Re-identification: A Survey and Outlook
        pos_idx = np.where(raw_cmc == 1)
        pos_max_idx = np.max(pos_idx)
        inp = cmc[pos_max_idx] / (pos_max_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)

    return all_cmc, mAP, mINP




def eval_llcm(distmat, q_pids, g_pids, q_camids, g_camids, max_rank = 20):
    """Evaluation with sysu metric
    Key: for each query identity, its gallery images from the same camera view are discarded. "Following the original setting in ite dataset"
    """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    pred_label = g_pids[indices]
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    
    # compute cmc curve for each query
    new_all_cmc = []
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0. # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # the cmc calculation is different from standard protocol
        # we follow the protocol of the author's released code
        new_cmc = pred_label[q_idx][keep]
        new_index = np.unique(new_cmc, return_index=True)[1]

        new_cmc = [new_cmc[index] for index in sorted(new_index)]
        
        new_match = (new_cmc == q_pid).astype(np.int32)
        new_cmc = new_match.cumsum()
        new_all_cmc.append(new_cmc[:max_rank])
        
        orig_cmc = matches[q_idx][keep] # binary vector, positions with value 1 are correct matches
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()

        # compute mINP
        # refernece Deep Learning for Person Re-identification: A Survey and Outlook
        pos_idx = np.where(orig_cmc == 1)
        pos_max_idx = np.max(pos_idx)
        inp = cmc[pos_max_idx]/ (pos_max_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"
    
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q   # standard CMC
    
    new_all_cmc = np.asarray(new_all_cmc).astype(np.float32)
    new_all_cmc = new_all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)
    return new_all_cmc, mAP, mINP