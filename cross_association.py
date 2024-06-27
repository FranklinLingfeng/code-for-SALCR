import time
import numpy as np
import torch
import torch.nn.functional as F
import collections
import copy
from utils.rerank import cross_jaccard_dist
from utils.faiss_rerank import compute_jaccard_distance
from sklearn.cluster import DBSCAN


def pairwise_distance(x, y):
   
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist_mat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(x, y.t(), beta=1, alpha=-2)
        
    return dist_mat



def generate_one_hot_label(labels, num_cls):
    num_instance = labels.shape[0]
    one_hot_labels = torch.zeros(num_instance, num_cls)
    one_hot_order = torch.arange(num_instance)
    one_hot_labels[one_hot_order, labels] = 1
    
    return one_hot_labels


def generate_center(pseudo_labels, features):
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


def sinkhorn_solver(P, lambda_sk=5, max_iter=500):
    
    tt = time.time()
    
    num_instance = P.shape[0]
    num_clusters = P.shape[1]
    
    alpha = torch.ones((num_instance, 1)) / num_instance  # initial value for alpha
    beta = torch.ones((num_clusters, 1)) / num_clusters  # initial value for beta
    
    inv_K = 1. / num_instance
    inv_N = 1. / num_clusters   
    PS = torch.exp(-lambda_sk * P)
        
    err = 1e6
    step = 0  
    
    while err > 1e-1 and step < max_iter:
        alpha = inv_K / (torch.mm(PS, beta))  # (KxN) @ (N,1) = K x 1
        beta_new = inv_N / (torch.mm(alpha.t(), PS)).T  # ((1,K) @ (KxN)).t() = N x 1
        if step % 10 == 0:
            err = np.nansum(np.abs(beta / beta_new - 1))
        beta = beta_new
        step += 1
        
    print("Sinkhorn-Knopp   Error: {:.3f}   Total step: {}   Total time: {:.3f}".format(err, step, time.time() - tt))
    P_out = torch.diag(alpha.squeeze()).mm(PS).mm(torch.diag(beta.squeeze()))
        
    return P_out


def lp(aff_ir, aff_rgb, aff_ir_rgb, aff_rgb_ir, y_rgb, y_ir_cm, alpha):

    tt = time.time()
    num_cls = y_rgb.shape[1]

    ## init
    y_rgb_old = copy.deepcopy(y_rgb)
    y_ir_cm_old = copy.deepcopy(y_ir_cm)
    y_rgb_new = copy.deepcopy(y_rgb_old)
    y_ir_cm_new = copy.deepcopy(y_ir_cm_old)
    eps = 1e6; iter = 0; max_iter = 20

    while eps > 1e-2 and iter < max_iter:
        
        y_rgb_new = (1 - alpha) * aff_rgb_ir.mm(y_ir_cm_new) + alpha * y_rgb
        y_ir_cm_new = (1 - alpha) * aff_ir_rgb.mm(y_rgb_new) + alpha * y_ir_cm
        y_rgb_new = 0.5 * aff_rgb.mm(y_rgb_new) + 0.5 * y_rgb_new
        y_ir_cm_new = 0.5 * aff_ir.mm(y_ir_cm_new) + 0.5 * y_ir_cm_new

        eps = torch.max((y_ir_cm_new - y_ir_cm_old).abs().sum() / y_ir_cm_old.sum(), 
                (y_rgb_new - y_rgb_old).abs().sum() / y_rgb_old.sum())
        y_ir_cm_old = copy.deepcopy(y_ir_cm_new)
        y_rgb_old = copy.deepcopy(y_rgb_new)
        iter += 1

        print('Total time for label propagation: {:.5f} eps: {:.5f} iter: {:3d}'.format(time.time() - tt, eps, iter))

    return y_ir_cm_new, y_rgb_new



def MULT(args, epoch, features_RGB, features_IR, labels_RGB, labels_IR, centers_RGB, centers_IR, dist_rgb, dist_ir, temp=0.05):

    N_rgb = features_RGB.shape[0]
    N_ir = features_IR.shape[0]
    num_cls_rgb = centers_RGB.shape[0]
    num_cls_ir = centers_IR.shape[0]

    ## generate one hot label / init cross-modality labels
    y_rgb = torch.softmax(features_RGB.mm(centers_RGB.t()) / temp, dim=1)
    y_ir = torch.softmax(features_IR.mm(centers_IR.t()) / temp, dim=1)
    y_rgb_cm = generate_one_hot_label(sinkhorn_solver(pairwise_distance(features_RGB, centers_IR)).argmax(1), num_cls_ir)
    y_ir_cm = generate_one_hot_label(sinkhorn_solver(pairwise_distance(features_IR, centers_RGB)).argmax(1), num_cls_rgb)

    ## compute affinity matrix for single/cross modality 
    print('Start constructing graph...')
    tt = time.time()
    aff_rgb = 1 - dist_rgb
    aff_ir = 1 - dist_ir

    ## normalization for affinity matrix
    aff_rgb = aff_rgb / aff_rgb.sum(1).unsqueeze(1)
    aff_ir = aff_ir / aff_ir.sum(1).unsqueeze(1)

    if args.dist == 'normal':
        P_cross = knn_graph(torch.softmax(features_RGB.mm(features_IR.t()), dim=1), k=30)
        P_cross_t = knn_graph(torch.softmax(features_IR.mm(features_RGB.t()), dim=1), k=30)
        aff_rgb_ir = P_cross / P_cross.sum(1).unsqueeze(1)
        aff_ir_rgb = P_cross_t / P_cross_t.sum(1).unsqueeze(1)

    elif args.dist == 'jaccard':
        # P_cross = sinkhorn_solver(pairwise_distance(features_RGB, features_IR), lambda_sk=25)
        q_g_dist = np.array(pairwise_distance(features_RGB, features_IR))
        q_q_dist = np.array(pairwise_distance(features_RGB, features_RGB))
        g_g_dist = np.array(pairwise_distance(features_IR, features_IR))
        P_cross = cross_jaccard_dist(q_g_dist, q_q_dist, g_g_dist, k1=30, k2=6)
        aff_rgb_ir = P_cross / P_cross.sum(1).unsqueeze(1)
        aff_ir_rgb = P_cross.t() / P_cross.t().sum(1).unsqueeze(1)

    elif args.dist == 'OT':
        P_cross = sinkhorn_solver(pairwise_distance(features_RGB, features_IR), lambda_sk=25)
        aff_rgb_ir = P_cross / P_cross.sum(1).unsqueeze(1)
        aff_ir_rgb = P_cross.t() / P_cross.t().sum(1).unsqueeze(1)
    
    print('Total time for constructing graph: {:.5f}'.format(time.time() - tt))

    ## label propagation
    print('Value of alpha for label transfer: {:.2f}'.format(args.alpha))
    y_rgb_cm_new, y_ir_new = lp(aff_rgb, aff_ir, aff_rgb_ir, aff_ir_rgb, y_ir, y_rgb_cm, alpha=args.alpha)
    y_ir_cm_new, y_rgb_new = lp(aff_ir, aff_rgb, aff_ir_rgb, aff_rgb_ir, y_rgb, y_ir_cm, alpha=args.alpha)

    print(y_rgb_new.argmax(1).unique().shape[0], y_ir_cm_new.argmax(1).unique().shape[0],\
        y_ir_new.argmax(1).unique().shape[0], y_rgb_cm_new.argmax(1).unique().shape[0])

    labels_RGB = y_rgb_new.argmax(1)
    labels_IR = y_ir_new.argmax(1)
    RGB_instance_IR_label = y_rgb_cm_new.argmax(1)
    IR_instance_RGB_label = y_ir_cm_new.argmax(1)

    return labels_RGB, labels_IR, RGB_instance_IR_label, IR_instance_RGB_label, centers_RGB, centers_IR,\
            y_rgb_new, y_ir_new, y_rgb_cm_new, y_ir_cm_new




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
    # print('----number of outliers: {:5d}'.format(num_outliers))

    return centers



def knn_graph(init_graph, k=30):

    ## select KNN neighbors
    threshold = init_graph.sort(dim=1, descending=True)[0][:, k-1]
    mask = torch.as_tensor(init_graph >= threshold.unsqueeze(1), dtype = torch.float)
    graph = mask * init_graph

    ## select similarities >= 0
    graph = graph.clamp(min=0)

    ## normalize
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



def part_ot(args, epoch, features_RGB_pg, features_IR_pg, labels_RGB, labels_IR, centers_RGB_pg, centers_IR_pg):

    ## prepare for association
    dim=2048
    part_num = args.part_num
    features_RGB, features_IR = F.normalize(features_RGB_pg[:, part_num*dim:], dim=-1), F.normalize(features_IR_pg[:, part_num*dim:], dim=-1)
    centers_RGB, centers_IR = generate_center(labels_RGB, features_RGB), generate_center(labels_IR, features_IR)

    features_RGB_p, features_IR_p = [], []
    for i in range(part_num):
        features_RGB_p.append(F.normalize(features_RGB_pg[:, i*dim : (i+1)*dim], dim=1).unsqueeze(0))
        features_IR_p.append(F.normalize(features_IR_pg[:, i*dim : (i+1)*dim], dim=1).unsqueeze(0))
    features_RGB_p, features_IR_p = torch.cat(features_RGB_p, 0), torch.cat(features_IR_p, 0)
    
    num_cls_rgb = centers_RGB_pg.shape[0]
    num_cls_ir = centers_IR_pg.shape[0]

    ## cross-modality affinities / feature fusion
    N_v, N_r = features_RGB.shape[0], features_IR.shape[0]

    if args.CMFP == True:
        A, features_ = agg_feature(features_RGB, features_IR, k=args.k_cmff)
        features_ = F.normalize(features_, dim=1)
    else:
        features_ = torch.cat((features_RGB, features_IR), dim=0)

    features_RGB_ = F.normalize(features_[:N_v], dim=1)
    features_IR_ = F.normalize(features_[N_v:], dim=1)
    centers_RGB_ = generate_center(labels_RGB, features_RGB_)
    centers_IR_ = generate_center(labels_IR, features_IR_)

    # optimal transport
    D_vr_, D_rv_ = pairwise_distance(features_RGB_, centers_IR_), pairwise_distance(features_IR_, centers_RGB_)
    Q_vr, Q_rv = sinkhorn_solver(D_vr_, lambda_sk=args.lambda_ot), sinkhorn_solver(D_rv_, lambda_sk=args.lambda_ot)
    labels_RGB = features_RGB_.mm(centers_RGB_.t()).argmax(1)
    labels_IR = features_IR_.mm(centers_IR_.t()).argmax(1) 
    cm_labels_RGB = Q_vr.argmax(1)
    cm_labels_IR = Q_rv.argmax(1)

    y_rgb = generate_one_hot_label(labels_RGB, num_cls_rgb)
    y_ir = generate_one_hot_label(labels_IR, num_cls_ir)
    y_rgb_cm = generate_one_hot_label(cm_labels_RGB, num_cls_ir)
    y_ir_cm = generate_one_hot_label(cm_labels_IR, num_cls_rgb)

    labels_RGB = y_rgb.argmax(1)
    labels_IR = y_ir.argmax(1)
    cm_labels_RGB = y_rgb_cm.argmax(1)
    cm_labels_IR = y_ir_cm.argmax(1)

    return labels_RGB, labels_IR, cm_labels_RGB, cm_labels_IR, y_rgb, y_ir, y_rgb_cm, y_ir_cm,\
        y_rgb, y_ir, y_rgb_cm, y_ir_cm, [features_RGB_, features_IR_, centers_RGB_, centers_IR_]



def PGM_matching(args, epoch, features_RGB, features_IR, labels_RGB, labels_IR, centers_RGB, centers_IR, dist_rgb, dist_ir):
    
    from scipy.optimize import linear_sum_assignment

    num_cls_rgb = centers_RGB.shape[0]
    num_cls_ir = centers_IR.shape[0]

    ######################## PGM
    print("Progressive Graph Matching")
    i2r = {}
    r2i = {}
    R = []
    bgm = False
    if num_cls_rgb >= num_cls_ir:
        # clusternorm
        cluster_features_rgb = F.normalize(centers_RGB, dim=1)
        cluster_features_ir = F.normalize(centers_IR, dim=1)
        # [-1, 1] torch.mm(cluster_features_rgb, cluster_features_ir.T) #CostMatrix
        similarity = ((torch.mm(cluster_features_rgb, cluster_features_ir.T))/1).exp().cpu() #.exp().cpu()
        dis_similarity = (1 / (similarity))
        cost = dis_similarity / 1 ### 809 * 411
        tmp = torch.zeros(dis_similarity.shape[0], dis_similarity.shape[0] - dis_similarity.shape[1])
        cost = (torch.cat((cost, tmp), 1)) ## 809 * 809
        unmatched_row = []
        row_ind, col_ind = linear_sum_assignment(cost)
        for idx, item in enumerate(row_ind):
            if col_ind[idx] < similarity.shape[1]:
                R.append((row_ind[idx], col_ind[idx]))
                r2i[row_ind[idx]] = col_ind[idx]
                i2r[col_ind[idx]] = row_ind[idx]
            else:
                unmatched_row.append(row_ind[idx])
        if bgm is False:
            unmatched_cost = cost[unmatched_row][:,:dis_similarity.shape[1]]
            unmatched_row_ind, unmatched_col_ind = linear_sum_assignment(unmatched_cost)
            for idx, item in enumerate(unmatched_row_ind):
                R.append((unmatched_row[idx], unmatched_col_ind[idx]))
                r2i[unmatched_row[idx]] = unmatched_col_ind[idx]
        del cluster_features_ir, cluster_features_rgb

    print("Progressive Graph Matching Done")
    ########################

    RGB_instance_IR_label = torch.zeros_like(labels_RGB)
    IR_instance_RGB_label = torch.zeros_like(labels_IR)
    for i, label in enumerate(labels_RGB):
        RGB_instance_IR_label[i] = r2i[label.item()]
    for i, label in enumerate(labels_IR):
        IR_instance_RGB_label[i] = i2r[label.item()]
        
    return labels_RGB, labels_IR, RGB_instance_IR_label, IR_instance_RGB_label, centers_RGB, centers_IR, 0, 0, 0, 0



def BCCM_matching(args, epoch, features_RGB, features_IR, labels_RGB, labels_IR, centers_RGB, centers_IR, dist_rgb, dist_ir):
    
    from scipy.optimize import linear_sum_assignment

    num_cls_rgb = centers_RGB.shape[0]
    num_cls_ir = centers_IR.shape[0]

    centers_RGB = F.normalize(centers_RGB, dim=1)
    centers_IR = F.normalize(centers_IR, dim=1)

    k = num_cls_rgb // num_cls_ir + 1
    cost_rgb_ir = pairwise_distance(centers_RGB, centers_IR).repeat(1, k)
    cost_ir_rgb = pairwise_distance(centers_IR, centers_RGB)
    
    RGB_instance_IR_label = torch.zeros_like(labels_RGB)
    IR_instance_RGB_label = torch.zeros_like(labels_IR)
    rgb_idx, ir_idx = linear_sum_assignment(cost_rgb_ir)
    for i, label in enumerate(labels_RGB):
        RGB_instance_IR_label[i] = ir_idx[label.item()] % num_cls_ir
    ir_idx, rgb_idx = linear_sum_assignment(cost_ir_rgb)
    for i, label in enumerate(labels_IR):
        IR_instance_RGB_label[i] = rgb_idx[label.item()]

    return labels_RGB, labels_IR, RGB_instance_IR_label, IR_instance_RGB_label, centers_RGB, centers_IR, 0, 0, 0, 0


def CLU(args, epoch, features_RGB, features_IR, labels_RGB, labels_IR, centers_RGB, centers_IR, dist_rgb, dist_ir):

    Homo_v = knn_graph(torch.softmax(features_RGB.mm(features_RGB.t()), dim=1), k=20)
    Homo_r = knn_graph(torch.softmax(features_IR.mm(features_IR.t()), dim=1), k=20)
    Hetero_vr = knn_graph(torch.softmax(features_RGB.mm(features_IR.t()), dim=1), k=20)
    Hetero_rv = knn_graph(torch.softmax(features_IR.mm(features_RGB.t()), dim=1), k=20)

    num_cls_rgb = centers_RGB.shape[0]
    num_cls_ir = centers_IR.shape[0]
    y_rgb = generate_one_hot_label(labels_RGB, num_cls_rgb)
    y_ir = generate_one_hot_label(labels_IR, num_cls_ir)

    RGB_instance_IR_label = Homo_v.mm(Hetero_vr.mm(y_ir)).argmax(1)
    IR_instance_RGB_label = Homo_r.mm(Hetero_rv.mm(y_rgb)).argmax(1)

    return labels_RGB, labels_IR, RGB_instance_IR_label, IR_instance_RGB_label, centers_RGB, centers_IR, 0, 0, 0, 0
        