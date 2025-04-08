import copy

import torch.nn as nn
import torch.nn.functional as F
from .mp_encoder import Mp_encoder, GConv
from .contrast import Contrast
import torch
import dgl
from copy import deepcopy
import random
from .normalize import normalize_adj_tensor
from torch_geometric.utils import subgraph
from torch_geometric.utils.map import map_index
from torch_geometric.utils.mask import index_to_mask
import torch.nn.init as init
from torch_geometric.utils import add_self_loops, to_dense_adj, dense_to_sparse, add_remaining_self_loops
import numpy as np
from .datasample import RandomNodeSamplingDataset, DataLoader

def dense_to_sparse_x(feat_index, n_node, n_dim):
    return torch.sparse.FloatTensor(feat_index,torch.ones(feat_index.shape[1]).to(feat_index.device),
                                    [n_node, n_dim])

def mask_edge(edge_index, p, weight=None):
    if p < 0. or p > 1.:
        raise ValueError(f'Mask probability has to be between 0 and 1 '
                         f'(got {p}')
    elif p == 0:
        return edge_index, weight
    else:
        e_ids = torch.arange(edge_index.size(1), dtype=torch.long, device=edge_index.device)
        mask = torch.full_like(e_ids, p, dtype=torch.float32)
        mask = torch.bernoulli(mask).to(torch.bool)
        return edge_index[:, ~mask], weight[~mask]

def drop_feature(x: torch.Tensor, p: float) -> torch.Tensor:
    if p < 0. or p > 1.:
        raise ValueError(f'Mask probability has to be between 0 and 1 '
                         f'(got {p}')
    elif p == 0:
        return x
    else:
        device = x.device
        drop_mask = torch.empty((x.size(1),), dtype=torch.float32).uniform_(0, 1) < p
        drop_mask = drop_mask.to(device)
        x = x.clone()
        x[:, drop_mask] = 0
    return x

def add_noise(x, noise_ratio=0.2, mean=0.0, std=0.1):
    device = x.device
    num_nodes = x.shape[0]
    num_noisy_nodes = int(num_nodes * noise_ratio)

    # 随机选择需要添加噪声的节点索引
    indices = torch.randperm(num_nodes)[:num_noisy_nodes]

    # 生成与选择的节点相同形状的高斯噪声
    mean = torch.mean(x)  # 计算整个特征矩阵的均值
    std = torch.var(x)  #
    noise = torch.normal(mean=mean, std=std, size=x[indices].shape).to(device)

    # 添加噪声到选择的节点特征
    augmented_features = x.clone()  # 创建一个副本以避免改变原始特征
    augmented_features[indices] += noise  # 仅对选择的节点添加噪声
    return augmented_features

def drop_node(x: torch.Tensor, p: float) -> torch.Tensor:
    if p < 0. or p > 1.:
        raise ValueError(f'Mask probability has to be between 0 and 1 '
                         f'(got {p}')
    elif p == 0:
        return x
    else:
        device = x.device
        drop_mask = torch.empty((x.size(0),), dtype=torch.float32).uniform_(0, 1) < p
        drop_mask = drop_mask.to(device)
        x = x.clone()
        x[drop_mask] = 0
    return x

class HGMS(nn.Module):
    def __init__(self, args, feats_dim_list, P):
        super(HGMS, self).__init__()
        self.P = P
        self.hidden_dim = args.hidden_dim
        self.fc_list = nn.ModuleList([nn.Linear(feats_dim, args.hidden_dim, bias=True)
                                      for feats_dim in feats_dim_list])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)

        self.mp = Mp_encoder(P, self.hidden_dim, args.MLP_layers, args.activate, args.feat_drop, args.attn_drop, feats_dim_list[0])
        self.thres = AdaptiveSoftThreshold(1)
        self.contrast = Contrast(self.hidden_dim, args.tau, args.lam)
        self.shrink = 1.0 / self.hidden_dim
        self.Self_Pressive_Network = Self_Pressive_Network(self.hidden_dim, P, args.express_layer, args.express_l2)

    def initialize_xavier(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 使用 Xavier 初始化
                nn.init.xavier_uniform_(m.weight, gain=1.414)
                # nn.init.zeros_(m.bias)  # 偏置初始化为零

    def initialize_kaiming(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')

    def get_edge_ind_dict(self, g, canonical_etypes):
        etypes = [i[1] for i in canonical_etypes]
        homo_g = dgl.to_homogeneous(g)
        ndata_Type = homo_g.ndata['_TYPE']  # homo_g.ndata['_ID'],
        etype_ntype = {nt[1]: (nt[0], nt[2]) for i, nt in enumerate(canonical_etypes)}
        ntype_count = torch.bincount(ndata_Type)
        ntype_ind = torch.cat([torch.zeros([1], dtype=int).cuda(), torch.cumsum(ntype_count, 0)], dim=0).cpu().tolist()
        n_ind_dict = {nt: (ntype_ind[i], ntype_ind[i + 1]) for i, nt in enumerate(g.ntypes)}
        e_ind_dict = {nt: (n_ind_dict[etype_ntype[nt][0]], n_ind_dict[etype_ntype[nt][1]]) for nt in etypes}
        return e_ind_dict

    def subgraph(self, subset, edge_index, edge_attr=None, relabel_nodes=False, num_nodes=None) :

        device = edge_index.device

        if isinstance(subset, (list, tuple)):
            subset = torch.tensor(subset, dtype=torch.long, device=device)

        if subset.dtype != torch.bool:
            node_mask = index_to_mask(subset, size=num_nodes)
        else:
            num_nodes = subset.size(0)
            node_mask = subset
            subset = node_mask.nonzero().view(-1)

        edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
        edge_index = edge_index[:, edge_mask]
        edge_attr = torch.ones(edge_index.shape[1]).to(device)

        if relabel_nodes:
            edge_index, _ = map_index(
                edge_index.view(-1),
                subset,
                max_index=num_nodes,
                inclusive=True,
            )
            edge_index = edge_index.view(2, -1)

            return edge_index, edge_attr

    def sub_he_graph(self, src, dst, edge_index, num_nodes, edge_attr, relabel_nodes, return_edge_mask=False):
        device = edge_index.device

        subset1, subset2 = torch.arange(*src), torch.arange(*dst)
        subset1 = torch.tensor(subset1, dtype=torch.long, device=device)
        subset2 = torch.tensor(subset2, dtype=torch.long, device=device)
        subset = torch.cat([subset1, subset2])

        node_mask1 = index_to_mask(subset1, size=num_nodes)
        node_mask2 = index_to_mask(subset2, size=num_nodes)

        edge_mask = node_mask1[edge_index[0]] & node_mask2[edge_index[1]]
        edge_index = edge_index[:, edge_mask]
        edge_attr = torch.ones(edge_index.shape[1]).to(device)

        if relabel_nodes:
            edge_index, _ = map_index(
                edge_index.view(-1),
                subset,
                max_index=num_nodes,
                inclusive=True,
            )
            edge_index = edge_index.view(2, -1)
        # dst 从0开始计数
        edge_index[1] = edge_index[1] - (src[1]-src[0])

        if return_edge_mask:
            return edge_index, edge_attr, edge_mask
        else:
            return edge_index, edge_attr

    def get_metapaths(self, edge_index, e_ind_dict, all_node_num, sp=False, add_selfloop=False, normalize=False):
        etypes = list(e_ind_dict.keys())
        metapath_list = []
        for i in etypes:
            src = e_ind_dict[i][0]
            dst = e_ind_dict[i][1]
            num_node = src[1]-src[0]
            sub_index, sub_weight = self.sub_he_graph(src=src, dst=dst, edge_index=edge_index, num_nodes=all_node_num,
                                      relabel_nodes=True, edge_attr=None)
            sub_g = torch.sparse_coo_tensor(indices=sub_index, values=sub_weight,
                                            size=[num_node, dst[1]-dst[0]])
            metapath = torch.matmul(sub_g, sub_g.T)
            if sp == True:
                if normalize:
                    metapath = self.normalize_adj(metapath, sparse=True)#.to_sparse_coo()
                    metapath_list.append(metapath)
                    if add_selfloop:
                        edge_index, edge_weight = metapath.indices(), metapath.values()
                        edge_index, edge_weight = add_self_loops(edge_index, edge_weight, num_nodes=num_node)
                        metapath = torch.sparse_coo_tensor(indices=edge_index, values=edge_weight, size=[num_node, num_node])
                        metapath_list.append(metapath)
                else:
                    metapath_list.append(metapath.to_sparse())
            else:
                metapath_list.append(metapath)
        return metapath_list

    def get_metapaths_dblp(self, edge_index, e_ind_dict, all_node_num, sp=False, add_selfloop=False, normalize=False):
        ''' ap, pt, pc ----> apa, apcpa, aptpa'''
        num_node = 4057
        etypes = list(e_ind_dict.keys())
        metapath_list = []

        src = e_ind_dict['ap'][0]
        dst = e_ind_dict['ap'][1]
        ap_index, ap_weight = self.sub_he_graph(src=src, dst=dst, edge_index=edge_index, num_nodes=all_node_num,
                                                  relabel_nodes=True, edge_attr=None)
        ap = torch.sparse_coo_tensor(indices=ap_index, values=ap_weight,
                                        size=[num_node, dst[1] - dst[0]])
        for i in etypes:
            if i == 'ap':
                sub_g = ap
                metapath = torch.matmul(sub_g, sub_g.T)
            else:
                src = e_ind_dict[i][0]
                dst = e_ind_dict[i][1]
                p__index, p__weight = self.sub_he_graph(src=src, dst=dst, edge_index=edge_index, num_nodes=all_node_num,
                                      relabel_nodes=True, edge_attr=None)
                p_ = torch.sparse_coo_tensor(indices=p__index, values=p__weight,
                                        size=[src[1] - src[0], dst[1] - dst[0]])
                ap_ = ap @ p_
                metapath = ap_ @ ap_.T

            metapath = metapath.cuda()
            if sp == True:
                if normalize:
                    metapath = self.normalize_adj(metapath, sparse=True)
                    metapath_list.append(metapath)
                    if add_selfloop:
                        edge_index, edge_weight = metapath.indices(), metapath.values()
                        edge_index, edge_weight = add_self_loops(edge_index, edge_weight, num_nodes=num_node)
                        metapath = torch.sparse_coo_tensor(indices=edge_index, values=edge_weight, size=[num_node, num_node])
                        metapath_list.append(metapath)
                else:
                    metapath_list.append(metapath.to_sparse())
            else:
                metapath_list.append(metapath)
        return metapath_list

    def get_metapaths_academic(self, edge_index, e_ind_dict, all_node_num, sp=False, add_selfloop=False, normalize=False):
        ''' ap, pt, pc ----> apa, apcpa, aptpa'''
        num_node = 28646
        etypes = list(e_ind_dict.keys())
        metapath_list = []

        src = e_ind_dict['author-paper'][0]
        dst = e_ind_dict['author-paper'][1]
        ap_index, ap_weight = self.sub_he_graph(src=src, dst=dst, edge_index=edge_index, num_nodes=all_node_num,
                                                  relabel_nodes=True, edge_attr=None)
        ap = torch.sparse_coo_tensor(indices=ap_index, values=ap_weight,
                                        size=[num_node, dst[1] - dst[0]])
        for i in etypes:
            if i == 'author-paper':
                sub_g = ap
                metapath = torch.matmul(sub_g, sub_g.T)
            else:
                src = e_ind_dict[i][0]
                dst = e_ind_dict[i][1]
                if i == 'cite':
                    node_range = torch.arange(*src).to(edge_index.device)
                    p__index, p__weight = self.subgraph(node_range, edge_index, relabel_nodes=True, num_nodes=all_node_num)
                    p_ = torch.sparse_coo_tensor(indices=p__index, values=p__weight,
                                                 size=[src[1] - src[0], dst[1] - dst[0]])
                    metapath = ap @ p_ @ ap.T
                else:
                    p__index, p__weight = self.sub_he_graph(src=src, dst=dst, edge_index=edge_index, num_nodes=all_node_num,
                                          relabel_nodes=True, edge_attr=None)
                    p_ = torch.sparse_coo_tensor(indices=p__index, values=p__weight,
                                            size=[src[1] - src[0], dst[1] - dst[0]])
                    ap_ = ap @ p_
                    metapath = ap_ @ ap_.T

            metapath = metapath.cuda()
            if sp == True:
                if normalize:
                    metapath = self.normalize_adj(metapath, sparse=True)
                    metapath_list.append(metapath)
                    if add_selfloop:
                        edge_index, edge_weight = metapath.indices(), metapath.values()
                        edge_index, edge_weight = add_self_loops(edge_index, edge_weight, num_nodes=num_node)
                        metapath = torch.sparse_coo_tensor(indices=edge_index, values=edge_weight, size=[num_node, num_node])
                        metapath_list.append(metapath)
                else:
                    metapath_list.append(metapath.to_sparse())
            else:
                metapath_list.append(metapath)
        return metapath_list

    def normalize_adj(self, adj, sparse=False):
        device = adj.device
        if sparse:
            rowsum = torch.sparse.sum(adj, dim=1).to_dense()  # Convert to dense to perform element-wise operations
            # Compute D^(-1/2)
            d_inv_sqrt = torch.pow(rowsum, -0.5)
            d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.  # Handle infinite values
            # Create diagonal matrix D^(-1/2) as a sparse tensor
            d_mat_inv_sqrt = torch.sparse_coo_tensor(torch.arange(adj.size(0)).unsqueeze(0).repeat(2, 1).to(device),
                                                     d_inv_sqrt,
                                                     size=adj.shape,
                                                     dtype=adj.dtype)
            # Perform normalization: D^(-1/2) * A * D^(-1/2)
            adj_normalized = torch.sparse.mm(d_mat_inv_sqrt, adj)
            adj_normalized = torch.sparse.mm(adj_normalized, d_mat_inv_sqrt)
            return adj_normalized
        else:
            """Symmetrically normalize adjacency matrix."""
            # Calculate the sum of each row (degree)
            rowsum = adj.sum(1)
            # Compute D^(-1/2)
            d_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
            d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.  # Handle infinite values
            # Create diagonal matrix D^(-1/2)
            d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
            # Return the normalized adjacency matrix
            return d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt #torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)

    def hg_random_aug(self, args, hg, ori_g, x, canonical_etypes, sp, add_selfloop, normalize):
        node_num = ori_g.shape[0]
        e_ind_dict = self.get_edge_ind_dict(hg, canonical_etypes)
        # 特征增广

        x1, x2 = drop_feature(x, args.feat_mask_ratio1), drop_feature(x, args.feat_mask_ratio2)

        # 结构增广
        edges, weight = ori_g.indices(), ori_g.val
        # edges, weight = ori_g.coalesce().indices(), ori_g.coalesce().values()
        edge_index1, edge_weight1 = mask_edge(edge_index=edges, weight=weight, p=args.edge_mask_ratio1)  # edges, weight
        edge_index2, edge_weight2 = mask_edge(edge_index=edges, weight=weight, p=args.edge_mask_ratio2)

        if args.dataset == 'dblp':
            mps1, mps2 = self.get_metapaths_dblp(edge_index1, e_ind_dict, all_node_num=node_num, sp=sp, add_selfloop=add_selfloop, normalize=args.use_normalize),\
                         self.get_metapaths_dblp(edge_index2, e_ind_dict, all_node_num=node_num, sp=sp, add_selfloop=add_selfloop, normalize=args.use_normalize)
        elif args.dataset == 'academic':
            mps1, mps2 = self.get_metapaths_academic(edge_index1, e_ind_dict, all_node_num=node_num, sp=sp, add_selfloop=add_selfloop, normalize=args.use_normalize),\
                         self.get_metapaths_academic(edge_index2, e_ind_dict, all_node_num=node_num, sp=sp, add_selfloop=add_selfloop, normalize=args.use_normalize)
        else:
            mps1, mps2 = self.get_metapaths(edge_index1, e_ind_dict, all_node_num=node_num, sp=sp, add_selfloop=add_selfloop, normalize=args.use_normalize),\
                         self.get_metapaths(edge_index2, e_ind_dict, all_node_num=node_num, sp=sp, add_selfloop=add_selfloop, normalize=args.use_normalize)

        return x1, x2, mps1, mps2

    def mp_random_aug(self, args, mps, x, add_selfloop=False):
        # 特征增广
        x1, x2 = drop_feature(x, args.feat_mask_ratio1), drop_feature(x, args.feat_mask_ratio2)
        # 结构增广
        mps1, mps2 = [], []
        for mp in mps:
            n_num = mp.shape[0]
            edge_index = mp.coalesce().indices()
            edge_weights = mp.coalesce().values()
            # 先移除自连接边
            mask = edge_index[0] == edge_index[1]
            self_edges = edge_index[:, mask]
            self_weights = edge_weights[mask]

            edge_index = edge_index[:, ~mask]
            edge_weights = edge_weights[~mask]

            edge_index1, edge_weight1 = mask_edge(edge_index=edge_index, weight=edge_weights, p=args.edge_mask_ratio1)
            edge_index2, edge_weight2 = mask_edge(edge_index=edge_index, weight=edge_weights, p=args.edge_mask_ratio2)

            # if add_selfloop == True:
            #     edge_index1, edge_weight1 = add_self_loops(edge_index1, edge_weight1, num_nodes=n_num)
            #     edge_index2, edge_weight2 = add_self_loops(edge_index2, edge_weight2, num_nodes=n_num)
            edge_index1, edge_weight1 = torch.cat([edge_index1, self_edges], dim=1), torch.cat([edge_weight1, self_weights], dim=0)
            edge_index2, edge_weight2 = torch.cat([edge_index2, self_edges], dim=1), torch.cat([edge_weight2, self_weights], dim=0)

            mp1 = torch.sparse_coo_tensor(indices=edge_index1, values=edge_weight1, size=[n_num, n_num])
            mp2 = torch.sparse_coo_tensor(indices=edge_index2, values=edge_weight2, size=[n_num, n_num])

            # mp1 = self.normalize_adj(mp1, sparse=True)
            # mp2 = self.normalize_adj(mp2, sparse=True)

            mps1.append(mp1)
            mps2.append(mp2)
        return x1, x2, mps1, mps2

    def remove_self_loops(self, edge_index, edge_weights):
        mask = edge_index[0] != edge_index[1]
        return edge_index[:, mask], edge_weights[mask]

    def mp_pathsim_aug(self, args, mps, pathsims, mask_ratio):
        aug_mps = []
        for i in range(len(mps)):
            mp, pathsim = mps[i], pathsims[i]
            n_num = mp.shape[0]
            edge_indices = mp.coalesce().indices()
            edge_weights = mp.coalesce().values()

            prob_matrix = pathsim[edge_indices[0], edge_indices[1]] # pathsim
            # prob_matrix = edge_weights # edge_weights
            # 将自连接边的概率设置为0
            mask = edge_indices[0] == edge_indices[1]
            prob_matrix[mask] = 1.0
            # 将权重设置为概率
            if prob_matrix.min() == 1: prob_matrix = prob_matrix
            else: prob_matrix = 1 - prob_matrix

            # 计算需要丢弃的边数
            num_edges = edge_indices.shape[1] - mask.sum()
            num_edges_to_drop = int(num_edges * mask_ratio)
            # 根据权重归一化的概率随机选择要丢弃的边
            drop_indices = torch.multinomial(prob_matrix, num_edges_to_drop, replacement=False)

            # 生成掩码，默认全为1
            mask = torch.ones(edge_indices.shape[1], dtype=torch.bool)

            # 将掩码中选中的边置为0（即丢弃这些边）
            mask[drop_indices] = 0

            # 根据掩码生成新的邻接矩阵
            indices = edge_indices[:, mask]
            values = edge_weights[mask]
            aug_mp = torch.sparse_coo_tensor(indices=indices, values=values, size=[n_num, n_num])
            aug_mp = self.normalize_adj(aug_mp, sparse=True)
            aug_mps.append(aug_mp)
        return aug_mps

    def mutiview_closed_selfpress(self, XS, att_beta, K, P, alpha1, alpha2, beta):
        '''∑||X_v — SX_v|| + β ∑ (s_ij^2) + α1 ||S-K|| + ||S-P||'''
        coe2 = 1.0 / (beta + alpha1 + alpha2)

        # X_cat = torch.cat(XS, dim=1)

        # attention weight
        X_cat = []
        for X, beta in zip(XS, att_beta):
            X_cat.append(torch.sqrt(beta) * X)  # 每个矩阵与对应的权重相乘 * len(XS)
        X_cat = torch.cat(X_cat, dim=1)  # 合并所有加权矩阵

        inv = torch.inverse(torch.eye(X_cat.shape[1]).to(X_cat.device) + (coe2 * X_cat.T @ X_cat))  # Q中的逆矩阵
        res = X_cat @ inv @ X_cat.T  # B中第二项的后面一部分
        res = coe2 * torch.eye(X_cat.shape[0]).to(X_cat.device) - coe2 * coe2 * res  # B的完整计算
        S = (X_cat @ X_cat.T + alpha1 * K + alpha2 * P) @ res
        return S


    def S_normal(self, S):
        # 最大最小归一化
        # min_val = S.min()
        # max_val = S.max()
        min_val = S.min(dim=1, keepdim=True)[0]  # 获取每一行的最小值
        max_val = S.max(dim=1, keepdim=True)[0]  # 获取每一行的最大值
        S = (S - min_val) / (max_val - min_val)
        S = (torch.abs(S) + torch.abs(S.T)) / 2
        return S

    def get_coeff(self, q_emb, k_emb):
        c = self.thres(q_emb.mm(k_emb.t()))
        return self.shrink * c

    def regularizer(self, c, lmbd=1.0):
        return lmbd * torch.abs(c).sum() + (1.0 - lmbd) / 2.0 * torch.pow(c, 2).sum()

    def forward(self, args, hg, ori_g, x, mps, S, topk_graph, pathsims, pos, canonical_etypes, epoch):  # p a s
        N, M = x.shape[0], len(mps)
        '''heterogeous graph augmentation'''
        x1, x2, mps1, mps2 = self.hg_random_aug(args, hg, ori_g, x, canonical_etypes, sp=True, add_selfloop=False, normalize=args.use_normalize)

        '''mp random augmentation'''
        # x1, x2, mps1, mps2 = self.mp_random_aug(args, mps, x, add_selfloop=True)

        '''mp pathsim augmentation'''
        # x1, x2 = drop_feature(x, args.feat_mask_ratio1), drop_feature(x, args.feat_mask_ratio2)
        # mps1, mps2 = self.mp_pathsim_aug(args, mps, pathsims, args.edge_mask_ratio1),\
        #              self.mp_pathsim_aug(args, mps, pathsims, args.edge_mask_ratio2)

        # encoder
        z1, zs1, h, _ = self.mp(x1, mps1, fc=True, return_list=True)
        z2, zs2, _, _ = self.mp(x2, mps2, fc=True, return_list=True)
        with torch.no_grad():
           z, zs, _, beta = self.mp(x, mps, fc=True, return_list=True)

        if args.self_expressive == 'network':
            '''self-expressive network'''
            self_loss, S = self.Self_Pressive_Network(args, zs, mps, beta, topk_graph, pos)
            S = self.S_normal(S)
            S_ = S.clone().detach()
            sampled_S = S[torch.randperm(S_.size(0))[:2000]][:, torch.randperm(S_.size(0))[:2000]]
            threshold = torch.quantile(sampled_S, args.quantile1)  # , dim=1
            S = torch.where(S >= threshold, S, 0.0)  # .unsqueeze(1)
            S.fill_diagonal_(1)
            z_S = S.detach() @ h
        elif args.self_expressive == 'closed':
            ''' closed-form self-expressive'''
            self_loss = 0.0
            S = self.mutiview_closed_selfpress(zs, beta, topk_graph, pos, alpha1=args.alpha1, alpha2=args.alpha2, beta=args.beta)
            sampled_S = S[torch.randperm(S.size(0))[:2000]][:, torch.randperm(S.size(0))[:2000]]
            outlier = torch.quantile(sampled_S, args.outlier)  # , dim=1
            threshold = torch.quantile(sampled_S, args.quantile1)
            S = torch.where(S <= outlier, S, 0.0)
            S = torch.where(S >= threshold, S, 0.0)
            S_ = S.clone()
            S_ = self.S_normal(S_)
            # print(S_)
            z_S = S @ h

        ''' 创建 S 视图'''
        if args.use_pos: pass
        else: pos = torch.eye(pos.size(0)).to(args.device)
        if args.interval == 0.0:
            gamma = args.gamma
        else:
            gamma = min(0.0 + epoch * args.interval, args.gamma)  #
        if args.dataset == 'academic':
            node_mask = torch.ones((z.size(0),)).bool().to(args.device)
            nodes = RandomNodeSamplingDataset(node_mask)
            node_loader = DataLoader(nodes, batch_size=128, shuffle=True, num_workers=0)
            loss = 0.0
            for node_index in node_loader:
                node_index = node_index.to(args.device)
                pos = pos.to_dense()
                loss += self.contrast(z1[node_index], z2[node_index], pos[node_index][:, node_index])
                loss += self.contrast(z1[node_index], z2[node_index], pos[node_index][:, node_index], S_[node_index][:, node_index], args.quantile2) \
                        + self.contrast(z1[node_index], z_S[node_index], pos[node_index][:, node_index], S_[node_index][:, node_index], args.quantile2) * gamma
            loss = loss/len(node_loader)
        else:
            loss = self.contrast(z1, z2, pos, S_, args.quantile2) + self.contrast(z1, z_S, pos, S_, args.quantile2) * gamma
        return (loss, self_loss), S_, mps

    def get_embeds(self, feats, mps, fc=False, detach=True):
        if fc == True:
            z_mp = feats
        else:
            z_mp = F.elu(self.fc_list[0](feats[0]))
        z_mp, z_list, _, _ = self.mp(z_mp, mps, fc=True, return_list=True)

        if detach == True:
            return z_mp.detach(), [i.detach() for i in z_list]
        else:
            return z_mp

class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers,
                 dropout, activate, last_activate=True):
        super(MLP, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels, bias=True))
        self.bns = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))

        self.bn0 = nn.BatchNorm1d(in_channels)
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 1):
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        if activate == 'elu':
            self.activate = nn.ELU()
        elif activate == 'relu':
            self.activate = nn.ReLU()
        elif activate == 'leaky_relu':
            self.activate = nn.LeakyReLU()
        elif activate == 'gelu':
            self.activate = nn.GELU()
        else:
            self.activate = lambda x: x

        self.dropout = dropout
        self.last_activate = last_activate

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x):
        for lin, bn in zip(self.lins[:-1], self.bns[:-1]):
            x = lin(x)
            x = self.activate(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        if self.last_activate:
            x = self.activate(x)
        return x

class Self_Pressive_Network(nn.Module):
    def __init__(self, hidden_dim, P, express_layer, express_l2):
        super(Self_Pressive_Network, self).__init__()
        self.express_mlp = nn.ModuleList([Expressive_MLP(input_dims=hidden_dim, hid_dims=[hidden_dim] * express_layer, \
                                                         out_dims=hidden_dim, kaiming_init=True) for _ in range(P)])
        self.thres = AdaptiveSoftThreshold(1)
        self.shrink = 1.0 / hidden_dim
        self.express_l2 = express_l2

    def get_coeff(self, q_emb, k_emb):
        c = self.thres(q_emb.mm(k_emb.t()))
        return self.shrink * c

    def regularizer(self, c, lmbd=1.0):
        return lmbd * torch.abs(c).sum() + (1.0 - lmbd) / 2.0 * torch.pow(c, 2).sum()

    def forward(self, args, zs, mps, beta, K, P):
        N, M = zs[0].shape[0], len(mps)
        # self-pressive network
        rec_loss, reg_loss = 0, 0
        C = torch.zeros([M, N, N]).to(args.device)
        for i in range(M):
            z = zs[i].clone().detach()
            express_z = self.express_mlp[i](zs[i])
            C[i] = self.get_coeff(express_z, express_z)
            C[i] = C[i] - torch.diag(torch.diag(C[i]))
            c = C[i].detach()
            reg_loss += self.regularizer(c, args.lmbd)
            rec_loss += torch.sum(torch.pow(z - C[i] @ z, 2))
        # S = torch.mean(C, dim=0)
        S = 0
        for i in range(M):
            S += C[i] * beta[i]
        loss1 = torch.sum(torch.pow(S - K, 2))
        loss2 = torch.sum(torch.pow(S - P, 2))
        loss = (rec_loss + reg_loss * args.express_l2 + loss1 * args.beta1 + loss2 * args.beta2) / N
        return loss, S

class Expressive_MLP(nn.Module):

    def __init__(self, input_dims, hid_dims, out_dims, kaiming_init=False):
        super(Expressive_MLP, self).__init__()
        self.input_dims = input_dims
        self.hid_dims = hid_dims
        self.output_dims = out_dims
        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(self.input_dims, self.hid_dims[0]))
        self.layers.append(nn.ReLU())
        for i in range(len(hid_dims) - 1):
            self.layers.append(nn.Linear(self.hid_dims[i], self.hid_dims[i + 1]))
            self.layers.append(nn.ReLU())

        self.out_layer = nn.Linear(self.hid_dims[-1], self.output_dims)
        if kaiming_init:
            self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                init.kaiming_uniform_(layer.weight)
                init.zeros_(layer.bias)
        init.xavier_uniform_(self.out_layer.weight)
        init.zeros_(self.out_layer.bias)

    def forward(self, x):
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
        h = self.out_layer(h)
        h = torch.tanh_(h)
        return h

class AdaptiveSoftThreshold(nn.Module):
    def __init__(self, dim):
        super(AdaptiveSoftThreshold, self).__init__()
        self.dim = dim
        self.register_parameter("bias", nn.Parameter(torch.from_numpy(np.zeros(shape=[self.dim])).float()))

    def forward(self, c):
        return torch.sign(c) * torch.relu(torch.abs(c) - self.bias)