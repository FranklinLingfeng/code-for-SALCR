import numpy as np
import torch

def intra_modality_re_ranking(original_dist, k1=20, k2=6, lambda_value=0.3):
    all_num = original_dist.shape[0]
    gallery_num = original_dist.shape[0]  # gallery_num=all_num
    original_dist = np.transpose(original_dist / np.max(original_dist, axis=0))
    V = np.zeros_like(original_dist).astype(np.float16)
    initial_rank = np.argsort(original_dist).astype(np.int32)  ## default axis=-1.

    print('Starting intra_modality re_ranking...')
    for i in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i,
                                :k1 + 1]  ## k1+1 because self always ranks first. forward_k_neigh_index.shape=[k1+1].  forward_k_neigh_index[0] == i.
        backward_k_neigh_index = initial_rank[forward_k_neigh_index,
                                 :k1 + 1]  ##backward.shape = [k1+1, k1+1]. For each ele in forward_k_neigh_index, find its rank k1 neighbors
        fi = np.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]  ## get R(p,k) in the paper
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate, :int(np.around(k1 / 2)) + 1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,
                                               :int(np.around(k1 / 2)) + 1]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * len(
                    candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)  ## element-wise unique
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = weight / np.sum(weight)
    # original_dist = original_dist[:query_num,]
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float16)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(gallery_num):
        invIndex.append(np.where(V[:, i] != 0)[0])  # len(invIndex)=all_num

    jaccard_dist = np.zeros_like(original_dist, dtype=np.float16)

    for i in range(all_num):
        temp_min = np.zeros(shape=[1, gallery_num], dtype=np.float16)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]],
                                                                               V[indImages[j], indNonZero[j]])
        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)

    pos_bool = (jaccard_dist < 0)
    jaccard_dist[pos_bool] = 0.0

    final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value

    return final_dist

def k_reciprocal_neigh(initial_rank, initial_rank_T, i, k1):
    forward_k_neigh_index = initial_rank[i, :k1 + 1]
    backward_k_neigh_index = initial_rank_T[forward_k_neigh_index, :k1 + 1]
    fi = np.where(backward_k_neigh_index == i)[0]
    return forward_k_neigh_index[fi]


def cal_v(original_dist, query_query_rank, query_gallery_rank, gallery_query_rank, gallery_gallery_rank, k1=20, k2=6):
    query_num = original_dist.shape[0]
    V = np.zeros_like(original_dist).astype(np.float16)

    print('Starting re_ranking...')
    for i in range(query_num):
        # k-reciprocal neighbors
        k_reciprocal_index = k_reciprocal_neigh(query_gallery_rank, gallery_query_rank, i, k1)
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_k_reciprocal_index = k_reciprocal_neigh(gallery_gallery_rank, gallery_gallery_rank, candidate,
                                                              int(np.around(k1 / 2)))
            if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2. / 3 * len(
                    candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)  ## element-wise unique
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = weight / np.sum(weight)

    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float32)
        for i in range(query_num):
            V_qe[i, :] = np.mean(V[query_query_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe

    return V


def cal_jaccard_dist(V):
    all_num = V.shape[0]
    invIndex = []
    for i in range(all_num):
        invIndex.append(np.where(V[:, i] != 0)[0])  # len(invIndex)=all_num
    jaccard_dist = np.zeros_like(V, dtype=np.float16)
    for i in range(all_num):
        temp_min = np.zeros(shape=[1, all_num], dtype=np.float16)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]],
                                                                               V[indImages[j], indNonZero[j]])
        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)

    return jaccard_dist

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

def cal_feat_mean(features,rank,k=20):
    features_res = torch.zeros((features.shape[0],features.shape[1]))
    for i in range(features.shape[0]):
        features_res[i] = torch.mean(torch.tensor(features[rank[i,:k],:]),dim = 0)
    return features_res
def cal_mean_feat_distmat(features_ir,features_rgb,indices_ir_ir,indices_rgb_rgb,k=20):
    features_ir_mean =  cal_feat_mean(features_ir,indices_ir_ir,k=k)
    features_rgb_mean =  cal_feat_mean(features_rgb,indices_rgb_rgb,k=k)
    distmat_ir_rgb_mean = cal_distmat(features_ir_mean,features_rgb_mean)
    distmat_ir_rgb_mean_norm =  distmat_ir_rgb_mean/np.max(distmat_ir_rgb_mean,axis = 0)
    return distmat_ir_rgb_mean_norm

def original_dist_pool(cross_features_ir,cross_features_rgb,distmat_ir_ir_rerank,distmat_rgb_rgb_rerank,k):
    indices_ir_ir_rerank = np.argsort(distmat_ir_ir_rerank, axis=1)
    indices_rgb_rgb_rerank = np.argsort(distmat_rgb_rgb_rerank, axis=1)
    distmat_ir_rgb_mean = cal_mean_feat_distmat(cross_features_ir, cross_features_rgb, indices_ir_ir_rerank,
                                                indices_rgb_rgb_rerank, k=k)
    return distmat_ir_rgb_mean

def re_ranking_cross(distmat_ir_rgb, distmat_rgb_ir, distmat_ir_ir, distmat_rgb_rgb, cross_features_ir, cross_features_rgb, k=20, eta_value=0.1):
    distmat_ir_rgb = distmat_ir_rgb / np.max(distmat_ir_rgb, axis=0)
    distmat_rgb_ir = distmat_rgb_ir / np.max(distmat_rgb_ir, axis=0)
    distmat_ir_ir = distmat_ir_ir / np.max(distmat_ir_ir, axis=0)
    distmat_rgb_rgb = distmat_rgb_rgb / np.max(distmat_rgb_rgb, axis=0)

    distmat_ir_ir_rerank = intra_modality_re_ranking(distmat_ir_ir, k1=k)
    distmat_rgb_rgb_rerank = intra_modality_re_ranking(distmat_rgb_rgb, k1=k)

    initial_rank_ir_rgb = np.argsort(distmat_ir_rgb, axis=1).astype(np.int32)
    initial_rank_rgb_ir = np.argsort(distmat_rgb_ir, axis=1).astype(np.int32)

    initial_rank_ir_ir = np.argsort(distmat_ir_ir_rerank, axis=1).astype(np.int32)
    initial_rank_rgb_rgb = np.argsort(distmat_rgb_rgb_rerank, axis=1).astype(np.int32)

    v_ir_ir = cal_v(distmat_ir_ir, initial_rank_ir_ir, initial_rank_ir_ir, initial_rank_ir_ir, initial_rank_ir_ir, k,
                    k)
    v_rgb_rgb = cal_v(distmat_rgb_rgb, initial_rank_rgb_rgb, initial_rank_rgb_rgb, initial_rank_rgb_rgb,
                      initial_rank_rgb_rgb, k, k)
    v_ir_rgb = cal_v(distmat_ir_rgb, initial_rank_ir_ir, initial_rank_ir_rgb, initial_rank_rgb_ir, initial_rank_rgb_rgb,
                     k, k)
    v_rgb_ir = cal_v(distmat_rgb_ir, initial_rank_rgb_rgb, initial_rank_rgb_ir, initial_rank_ir_rgb, initial_rank_ir_ir,
                     k, k)

    v_all = np.concatenate(
        [np.concatenate([v_ir_ir, v_ir_rgb], axis=1),
         np.concatenate([v_rgb_ir, v_rgb_rgb], axis=1)],
        axis=0)

    v_all_norm = v_all / np.sum(v_all, axis=1, keepdims=True)

    print('cal jaccard_dist')
    jaccard_dist = cal_jaccard_dist(v_all_norm)

    ir_num = distmat_ir_rgb.shape[0]
    jaccard_dist_ir_rgb = jaccard_dist[:ir_num, ir_num:]

    # distmat_ir_rgb_mean = original_dist_pool(cross_features_ir, cross_features_rgb, distmat_ir_ir_rerank, distmat_rgb_rgb_rerank,
    #                                          k=k)
    distmat_ir_rgb_mean = cal_distmat(cross_features_ir, cross_features_rgb)
    final_dist_ir_rgb = jaccard_dist_ir_rgb * (1 - eta_value) + distmat_ir_rgb_mean * eta_value
    final_dist_rgb_ir = final_dist_ir_rgb.T


    return final_dist_ir_rgb, final_dist_rgb_ir