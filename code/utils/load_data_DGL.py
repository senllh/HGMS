import numpy as np
import scipy.sparse as sp
import torch
import torch as th
from sklearn.preprocessing import OneHotEncoder
import dgl
from torch_sparse import SparseTensor
from torch_geometric.utils import scatter, segment
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch import Tensor
from collections import defaultdict
import torch.nn.functional as F
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import pickle
from functools import reduce

def encode_onehot(labels):
    labels = labels.reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(labels)
    labels_onehot = enc.transform(labels).toarray()
    return labels_onehot

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = th.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = th.from_numpy(sparse_mx.data)
    shape = th.Size(sparse_mx.shape)
    return th.sparse.FloatTensor(indices, values, shape)

def cal_homophily(edge_index, label):
    # metapath = metapath.nonzero()
    label = label.argmax(1)
    edge_index = [edge_index.row, edge_index.col]
    # 移除自连接边
    # mask = edge_index[0] != edge_index[1]  # 找到源节点和目标节点不同的边
    # edge_index[0] = edge_index[0][mask]
    # edge_index[1] = edge_index[1][mask]

    src_labels = label[edge_index[0]]
    dst_labels = label[edge_index[1]]
    # 判断标签是否相同
    same_label = (src_labels == dst_labels)
    # 计算标签相同的比例作为homophily
    homophily = same_label.mean()
    return homophily

def get_inverse_id(index, num):
    return_list = []
    for i in range(num):
        if i not in index:
            return_list.append(i)
    return return_list

def EdgePerturb(edge_index, aug_ratio, src_num, dst_num):
    edge_index = torch.Tensor(edge_index)
    _, edge_num = edge_index.size()
    permute_num = int(edge_num * aug_ratio)

    src_unif = torch.ones(1, src_num)
    dst_unif = torch.ones(1, dst_num)

    # # 随机抽样
    unif = torch.ones(edge_num)
    remove_idx = unif.multinomial((permute_num), replacement=False)

    # add_src_idx = src_unif.multinomial(permute_num, replacement=True)
    add_src_idx = edge_index[0, remove_idx.unsqueeze(0)]
    add_dst_idx = dst_unif.multinomial(permute_num, replacement=True)
    add_edge_idx = torch.cat([add_src_idx, add_dst_idx], dim=0)
    keep_id = get_inverse_id(remove_idx.tolist(), edge_num)

    edge_index = torch.cat((edge_index[:, keep_id], add_edge_idx), dim=1)
    return edge_index.cpu().numpy()

def load_acm(ratio, type_num, return_hg=False):

    # The order of node types: 0 p 1 a 2 s
    A, P, S = 7167, 4019, 60
    path = "../data/acm/"
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)

    feat_p = sp.load_npz(path + "p_feat.npz")
    feat_a = sp.eye(type_num[1])
    feat_s = sp.eye(type_num[2])

    feat_p = th.FloatTensor(preprocess_features(feat_p))
    feat_a = th.FloatTensor(preprocess_features(feat_a))
    feat_s = th.FloatTensor(preprocess_features(feat_s))

    # 加载原始数据
    pap = sp.load_npz(path + "pap.npz")
    psp = sp.load_npz(path + "psp.npz")


    # 计算pathsim矩阵
    pathsim_matrices = pathsim([pap, psp])

    # a = cal_homophily(pap, label)

    nei_a = np.load(path + "nei_a.npy", allow_pickle=True)
    nei_s = np.load(path + "nei_s.npy", allow_pickle=True)


    pa = np.loadtxt(path + 'pa.txt', dtype=int)#.T
    ps = np.loadtxt(path + 'ps.txt', dtype=int)#.T
    # pa = EdgePerturb(pa, 0.2, P, A)
    # ps = EdgePerturb(ps, 0.2, P, S)
    # pa = np.loadtxt(path + 'pa_SHAC_gumbel_0.1.txt', dtype=int).T
    # ps = np.loadtxt(path + 'ps_SHAC_gumbel_0.1.txt', dtype=int).T

    pos = sp.load_npz(path + "pos.npz")

    pa_ = sp.coo_matrix((np.ones(pa.shape[0]),(pa[:,0], pa[:, 1])),shape=(P,A)).toarray()
    ps_ = sp.coo_matrix((np.ones(ps.shape[0]),(ps[:,0], ps[:, 1])),shape=(P,S)).toarray()

    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]

    label = th.FloatTensor(label)
    nei_a = [th.LongTensor(i) for i in nei_a]
    nei_s = [th.LongTensor(i) for i in nei_s]

    # 构建原始图
    hg = dgl.heterograph({
        ('paper', 'pa', 'author'): (pa.T[0], pa.T[1]),
        ('paper', 'ps', 'subject'): (ps.T[0], ps.T[1]),
        ('subject', 'sp', 'paper'): (ps.T[1], ps.T[0]),
        ('author', 'ap', 'paper'): (pa.T[1], pa.T[0]),
        # ('paper', 'pap', 'paper'): pap.nonzero(),
        # ('paper', 'psp', 'paper'): psp.nonzero(),
    })

    pos = sparse_mx_to_torch_sparse_tensor(pos)

    pap = sparse_mx_to_torch_sparse_tensor(normalize_adj(pap))
    psp = sparse_mx_to_torch_sparse_tensor(normalize_adj(psp))

    train = [th.LongTensor(i) for i in train]
    val = [th.LongTensor(i) for i in val]
    test = [th.LongTensor(i) for i in test]

    canonical_etypes = [('paper', 'pa', 'author'), ('paper', 'ps', 'subject')]
    main_type = 'paper'

    '''生成 train val test'''
    # classes = np.load(path + "labels.npy").astype('int32')
    # ratio = [1, 3, 5, 10]#
    # for i in ratio:
    #     c_train, c_val, c_test = [], [], []
    #     for c in range(3):
    #         c_id = np.where(classes == c)[0]
    #         np.random.shuffle(c_id)
    #         num = int(1000/3)
    #         c_train.append(c_id[:i])
    #         if c == 0 :
    #             c_val.append(c_id[i: num + i + 1])
    #             c_test.append(c_id[num + i + 1: num * 2 + i + 2])
    #         else:
    #             c_val.append(c_id[i: num + i])
    #             c_test.append(c_id[num + i: num * 2 + i])
    #
    #     train = np.concatenate((c_train[0], c_train[1], c_train[2]))
    #     val = np.concatenate((c_val[0], c_val[1], c_val[2]))
    #     test = np.concatenate((c_test[0], c_test[1], c_test[2]))
    #     np.save(path + 'train_' + str(i) + '.npy', train)
    #     np.save(path + 'val_' + str(i) + '.npy', val)
    #     np.save(path + 'test_' + str(i) + '.npy', test)


    return hg, canonical_etypes, main_type, [feat_p, feat_a, feat_s], [pap, psp], pathsim_matrices, pos, label, train, val, test

# def load_acm(ratio, type_num, return_hg=False):
#     # The order of node types: 0 p 1 a 2 s
#     # path = "../../data/acm/"
#     path = "../data/acm/"
#     # import os
#     # print(os.path.exists("../data"))
#     label = np.load(path + "labels.npy").astype('int32')
#     label = encode_onehot(label)
#
#     P, A, S = 4019, 7167, 60
#     pos_num = 5
#
#     feat_p = sp.load_npz(path + "p_feat.npz")
#     feat_a = sp.eye(type_num[1])
#     feat_s = sp.eye(type_num[2])
#
#     feat_p = th.FloatTensor(preprocess_features(feat_p))
#     feat_a = th.FloatTensor(preprocess_features(feat_a))
#     feat_s = th.FloatTensor(preprocess_features(feat_s))
#
#     # 加载原始数据
#     pap = sp.load_npz(path + "pap.npz")
#     psp = sp.load_npz(path + "psp.npz")
#     nei_a = np.load(path + "nei_a.npy", allow_pickle=True)
#     nei_s = np.load(path + "nei_s.npy", allow_pickle=True)
#
#     def generate_metapaths(pa, ps):
#         pa, ps = pa.T, ps.T
#         pa_ = sp.coo_matrix((np.ones(pa.shape[0]), (pa[:, 0], pa[:, 1])), shape=(P, A)).toarray()
#         ps_ = sp.coo_matrix((np.ones(ps.shape[0]), (ps[:, 0], ps[:, 1])), shape=(P, S)).toarray()
#
#         pap = np.matmul(pa_, pa_.T) > 0
#         pap = sp.coo_matrix(pap)
#         psp = np.matmul(ps_, ps_.T) > 0
#         psp = sp.coo_matrix(psp)
#         return pap, psp
#
#     def generate_pos(pap, psp):
#         pap = pap / pap.sum(axis=-1).reshape(-1, 1)
#         psp = psp / psp.sum(axis=-1).reshape(-1, 1)
#         all = (pap + psp).A.astype("float32")
#
#         pos = np.zeros((P, P))
#         k = 0
#         for i in range(len(all)):
#             one = all[i].nonzero()[0]
#             if len(one) > pos_num:
#                 oo = np.argsort(-all[i, one])
#                 sele = one[oo[:pos_num]]
#                 pos[i, sele] = 1
#                 k += 1
#             else:
#                 pos[i, one] = 1
#         pos = sp.coo_matrix(pos)
#         return pos
#
#     # #加载随机攻击的数据
#     # aug_ratio = 0.1
#     # pap = sp.load_npz(path + "pap_random_attack_"+str(aug_ratio)+".npz")
#     # psp = sp.load_npz(path + "psp_random_attack_"+str(aug_ratio)+".npz")
#     # nei_a = np.load(path + "nei_a_random_attack_"+str(aug_ratio)+".npy", allow_pickle=True)
#     # nei_s = np.load(path + "nei_s_random_attack_"+str(aug_ratio)+".npy", allow_pickle=True)
#
#     pa = np.loadtxt(path + 'pa.txt', dtype=int).T
#     ps = np.loadtxt(path + 'ps.txt', dtype=int).T
#     pos = sp.load_npz(path + "pos.npz")
#
#     if args.attack_model == 'clean':
#         pa = np.loadtxt(path + 'pa.txt', dtype=int).T
#         ps = np.loadtxt(path + 'ps.txt', dtype=int).T
#         pos = sp.load_npz(path + "pos.npz")
#
#     elif args.attack_model == 'random':
#         pa = np.loadtxt(path + 'pa_random_' + str(args.attack_ratio) + '.txt', dtype=int).T
#         ps = np.loadtxt(path + 'ps_random_' + str(args.attack_ratio) + '.txt', dtype=int).T
#         # pap, psp = generate_metapaths(pa, ps)
#         # pos = generate_pos(pap, psp)
#         # pap = sp.load_npz(path + "pap.npz")
#         # psp = sp.load_npz(path + "psp.npz")
#
#     elif args.attack_model == 'CLGA':
#         if args.attack_type == 'poisoning':
#             # pa = np.loadtxt(path + 'pa_CLGA_' + str(args.attack_ratio) + '.txt', dtype=int)
#             # ps = np.loadtxt(path + 'ps_CLGA_' + str(args.attack_ratio) + '.txt', dtype=int)
#
#             # pa = np.loadtxt(path + 'pa_embedding_0.1.txt', dtype=int)
#             # ps = np.loadtxt(path + 'ps_embedding_0.1.txt', dtype=int)
#
#             pa = np.loadtxt(path + 'pa_CLGA_rule_0.1.txt', dtype=int)
#             ps = np.loadtxt(path + 'ps_CLGA_rule_0.1.txt', dtype=int)
#
#             pap_, psp_ = generate_metapaths(pa, ps)
#             pos = generate_pos(pap_, psp_)
#
#         elif args.attack_type == 'evasion':
#             pa_ = np.loadtxt(path + 'pa_CLGA_' + str(args.attack_ratio) + '.txt', dtype=int)
#             ps_ = np.loadtxt(path + 'ps_CLGA_' + str(args.attack_ratio) + '.txt', dtype=int)
#             pap, psp = generate_metapaths(pa_, ps_)
#         # pos = generate_pos(pap, psp)
#         # node_drop_id = np.loadtxt(path + 'node_drop_id_CLGA_' + str(args.attack_ratio) + '.txt', dtype=int)
#         # feat_p[node_drop_id] = 0.0 #0.0 , feat_p.mean(dim=0)
#
#     train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
#     test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
#     val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]
#
#     label = th.FloatTensor(label)
#     nei_a = [th.LongTensor(i) for i in nei_a]
#     nei_s = [th.LongTensor(i) for i in nei_s]
#
#     # 构建原始图
#     hg = dgl.heterograph({
#         ('paper', 'pa', 'author'): (pa[0], pa[1]),
#         ('author', 'ap', 'paper'): (pa[1], pa[0]),
#         ('paper', 'ps', 'subject'): (ps[0], ps[1]),
#         ('subject', 'sp', 'paper'): (ps[1], ps[0]),
#     })
#
#     hg.nodes['paper'].data['h'] = feat_p
#     hg.nodes['author'].data['h'] = feat_a
#     hg.nodes['subject'].data['h'] = feat_s
#
#     pap = sparse_mx_to_torch_sparse_tensor(normalize_adj(pap))
#     psp = sparse_mx_to_torch_sparse_tensor(normalize_adj(psp))
#     pos = sparse_mx_to_torch_sparse_tensor(pos)
#
#     train = [th.LongTensor(i) for i in train]
#     val = [th.LongTensor(i) for i in val]
#     test = [th.LongTensor(i) for i in test]
#
#     canonical_etypes = [('paper', 'pa', 'author'), ('paper', 'ps', 'subject')]
#     main_type = 'paper'
#
#     if return_hg == True:
#         return hg, canonical_etypes, main_type, [feat_p, feat_a, feat_s], [pap, psp], pos, label, train, val, test
#     else:
#         return [nei_a, nei_s], [feat_p, feat_a, feat_s], [pap, psp], pos, label, train, val, test

def load_dblp(ratio, type_num, return_hg=False):
    # The order of node types: 0 a 1 p 2 c 3 t
    A, P, C, T = 4057, 14328, 20, 7723
    path = "../data/dblp/"
    # path = "../../data/dblp/"

    def generate_metapaths(pa, pc, pt):
        pa, pc, pt = pa.T, pc.T, pt.T
        pa_ = sp.coo_matrix((np.ones(pa.shape[0]), (pa[:, 0], pa[:, 1])), shape=(P, A)).toarray()
        pc_ = sp.coo_matrix((np.ones(pc.shape[0]), (pc[:, 0], pc[:, 1])), shape=(P, C)).toarray()
        pt_ = sp.coo_matrix((np.ones(pt.shape[0]), (pt[:, 0], pt[:, 1])), shape=(P, T)).toarray()

        apa = np.matmul(pa_.T, pa_) #> 0
        apa = sp.coo_matrix(apa)

        apc = np.matmul(pa_.T, pc_)
        apcpa = np.matmul(apc, apc.T)
        apcpa = sp.coo_matrix(apcpa)

        apt = np.matmul(pa_.T, pt_)
        aptpa = np.matmul(apt, apt.T)
        aptpa = sp.coo_matrix(aptpa)

        return apa, apcpa, aptpa

    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)
    nei_p = np.load(path + "nei_p.npy", allow_pickle=True)

    pa = np.loadtxt(path + 'pa.txt', dtype=int).T
    pc = np.loadtxt(path + 'pc.txt', dtype=int).T
    pt = np.loadtxt(path + 'pt.txt', dtype=int).T
    # p_num, a_num, c_num = 14328, 4057,

    feat_a = sp.load_npz(path + "a_feat.npz").astype("float32")
    feat_p = sp.eye(type_num[1])
    feat_t = sp.eye(type_num[2])
    feat_c = sp.eye(type_num[3])

    # apa = sp.load_npz(path + "apa.npz")
    # apcpa = sp.load_npz(path + "apcpa.npz")
    # aptpa = sp.load_npz(path + "aptpa.npz")
    apa, apcpa, aptpa = generate_metapaths(pa, pc, pt)
    pathsim_matrices = pathsim([apa, apcpa, aptpa])
    pos = sp.load_npz(path + "pos.npz")
    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]
    
    label = th.FloatTensor(label)
    nei_p = [th.LongTensor(i) for i in nei_p]
    feat_p = th.FloatTensor(preprocess_features(feat_p))
    feat_a = th.FloatTensor(preprocess_features(feat_a))
    feat_c = th.FloatTensor(preprocess_features(feat_c))
    feat_t = th.FloatTensor(preprocess_features(feat_t))

    hg = dgl.heterograph({
        ('paper', 'pa', 'author'): (pa[0], pa[1]),
        ('author', 'ap', 'paper'): (pa[1], pa[0]),
        ('paper', 'pc', 'conference'): (pc[0], pc[1]),
        ('conference', 'cp', 'paper'): (pc[1], pc[0]),
        ('paper', 'pt', 'term'): (pt[0], pt[1]),
        ('term', 'tp', 'paper'): (pt[1], pt[0]),
    })
    # hg = dgl.to_bidirected(hg)
    hg.nodes['paper'].data['h'] = feat_p
    hg.nodes['author'].data['h'] = feat_a
    hg.nodes['conference'].data['h'] = feat_c
    hg.nodes['term'].data['h'] = feat_t

    # metapath graph
    mp_g = dgl.heterograph({
        ('author', 'apa', 'author'): apa.nonzero(),
        # ('movie', 'mdm', 'movie'): mdm.nonzero(),
        # ('author', 'aps', 'subject'): aps.nonzero(),
    })

    apa = sparse_mx_to_torch_sparse_tensor(normalize_adj(apa))
    apcpa = sparse_mx_to_torch_sparse_tensor(normalize_adj(apcpa))
    aptpa = sparse_mx_to_torch_sparse_tensor(normalize_adj(aptpa))
    pos = sparse_mx_to_torch_sparse_tensor(pos)
    train = [th.LongTensor(i) for i in train]
    val = [th.LongTensor(i) for i in val]
    test = [th.LongTensor(i) for i in test]
    canonical_etypes = [('author', 'ap', 'paper'), ('paper', 'pt', 'term'), ('paper', 'pc', 'conference')]
    # metapaths = ['auther-paper-author', 'auther-paper-conference-paper-auther', 'auther-paper-term-paper-auther']
    main_type = 'author'
    '''save randomwalk prob'''

    # sum_mp = apa + apcpa + aptpa
    # sum_mp_index = sum_mp.coalesce().indices()
    # sum_mp = dgl.DGLGraph((sum_mp_index[0], sum_mp_index[1]))
    # prob_matrix = get_connection_prob_matrix(sum_mp, num_walks=10, restart_prob=0.15, walk_length=5)
    # torch.save(prob_matrix, path + 'randomwalk_20' + '.pkl')

    # prob_matrix_5 = torch.load(path + 'randomwalk_5' + '.pkl')
    # prob_matrix_10 = torch.load(path + 'randomwalk_10' + '.pkl')
    prob_matrix_20 = torch.load(path + 'randomwalk_20' + '.pkl')
    prob_matrix = prob_matrix_20

    if return_hg == True:
        return hg, canonical_etypes, main_type, [feat_a, feat_p], [apa, apcpa, aptpa], pathsim_matrices, pos, label, train, val, test
    else:
        return [nei_p], [feat_a, feat_p], [apa, apcpa, aptpa], pos, label, train, val, test


def load_aminer(ratio, type_num, return_hg=False):
    # The order of node types: 0 p 1 a 2 r

    P, A, R = 6564, 13329, 35890

    def generate_metapaths(pa, pr):
        pa, pr = pa.T, pr.T
        pa_ = sp.coo_matrix((np.ones(pa.shape[0]), (pa[:, 0], pa[:, 1])), shape=(P, A)).toarray()
        pr_ = sp.coo_matrix((np.ones(pr.shape[0]), (pr[:, 0], pr[:, 1])), shape=(P, R)).toarray()

        pap = np.matmul(pa_, pa_.T)
        pap = sp.coo_matrix(pap)
        prp = np.matmul(pr_, pr_.T)
        prp = sp.coo_matrix(prp)
        return pap, prp

    path = "../data/aminer/"
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)
    nei_a = np.load(path + "nei_a.npy", allow_pickle=True)
    nei_r = np.load(path + "nei_r.npy", allow_pickle=True)

    pa = np.loadtxt(path + 'pa.txt', dtype=int).T
    pr = np.loadtxt(path + 'pr.txt', dtype=int).T

    pa = EdgePerturb(pa, 0.2, P, A)
    pr = EdgePerturb(pr, 0.2, P, R)

    # Because none of P, A or R has features, we assign one-hot encodings to all of them.
    feat_p = sp.eye(type_num[0])
    feat_a = sp.eye(type_num[1])
    feat_r = sp.eye(type_num[2])
    # pap = sp.load_npz(path + "pap.npz")
    # prp = sp.load_npz(path + "prp.npz")
    pos = sp.load_npz(path + "pos.npz")
    pap, prp = generate_metapaths(pa, pr)
    pathsim_matrices = pathsim([pap, prp])

    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]  #
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]

    label = th.FloatTensor(label)
    nei_a = [th.LongTensor(i) for i in nei_a]
    nei_r = [th.LongTensor(i) for i in nei_r]

    # pa, pr = torch.Tensor(pa), torch.Tensor(pr)
    feat_p = th.FloatTensor(preprocess_features(feat_p))
    feat_a = th.FloatTensor(preprocess_features(feat_a))
    feat_r = th.FloatTensor(preprocess_features(feat_r))
    # 构建原始图
    # node_num = type_num[0] + type_num[1] + type_num[2]
    # pa[1] = pa[1] + type_num[0]
    # pr[1] = pr[1] + type_num[0] + type_num[1]
    # ori_g = np.concatenate([pa, pr], axis=1)
    # ori_g = th.LongTensor(ori_g)
    # ori_g = th.sparse_coo_tensor(ori_g, th.ones(ori_g.size(1)), (node_num, node_num))#.to_dense()
    # edges_id = [(0, type_num[0], type_num[0], type_num[0] + type_num[1]),
    #            (0, type_num[0], type_num[0] + type_num[1], node_num)]

    #
    hg = dgl.heterograph({
        ('paper', 'pa', 'author'): (pa[0], pa[1]),
        ('author', 'ap', 'paper'): (pa[1], pa[0]),
        ('paper', 'pr', 'ref'): (pr[0], pr[1]),
        ('ref', 'rp', 'paper'): (pr[1], pr[0]),
    })
    # hg = dgl.to_bidirected(hg)
    hg.nodes['paper'].data['h'] = feat_p
    hg.nodes['author'].data['h'] = feat_a
    hg.nodes['ref'].data['h'] = feat_r

    pap = sparse_mx_to_torch_sparse_tensor(normalize_adj(pap))
    prp = sparse_mx_to_torch_sparse_tensor(normalize_adj(prp))
    pos = sparse_mx_to_torch_sparse_tensor(pos)
    train = [th.LongTensor(i) for i in train]
    val = [th.LongTensor(i) for i in val]
    test = [th.LongTensor(i) for i in test]

    canonical_etypes = [('paper', 'pa', 'author'), ('paper', 'pr', 'ref')]
    main_type = 'paper'
    if return_hg == True:

        return hg, canonical_etypes, main_type, [feat_p, feat_a, feat_r], [pap, prp], pathsim_matrices, pos, label, train, val, test
    else:
        return [nei_a, nei_r], [feat_p, feat_a, feat_r], [pap, prp], pos, label, train, val, test


def load_freebase(ratio, type_num, return_hg):
    M, A, D, W = 3492, 33401, 2502, 4459
    def generate_metapaths(ma, md, mw):
        ma, md, mw = ma.T, md.T, mw.T
        ma_ = sp.coo_matrix((np.ones(ma.shape[0]), (ma[:, 0], ma[:, 1])), shape=(M, A)).toarray()
        md_ = sp.coo_matrix((np.ones(md.shape[0]), (md[:, 0], md[:, 1])), shape=(M, D)).toarray()
        mw_ = sp.coo_matrix((np.ones(mw.shape[0]), (mw[:, 0], mw[:, 1])), shape=(M, W)).toarray()

        mam = np.matmul(ma_, ma_.T)
        mam = sp.coo_matrix(mam)
        mdm = np.matmul(md_, md_.T)
        mdm = sp.coo_matrix(mdm)
        mwm = np.matmul(mw_, mw_.T)
        mwm = sp.coo_matrix(mwm)
        return mam, mdm, mwm

    # The order of node types: 0 m 1 d 2 a 3 w
    path = "../data/freebase/"
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)
    nei_d = np.load(path + "nei_d.npy", allow_pickle=True)
    nei_a = np.load(path + "nei_a.npy", allow_pickle=True)
    nei_w = np.load(path + "nei_w.npy", allow_pickle=True)
    feat_m = sp.eye(M)
    feat_d = sp.eye(D)
    feat_a = sp.eye(A)
    feat_w = sp.eye(W)
    feat_m = th.FloatTensor(preprocess_features(feat_m))
    feat_d = th.FloatTensor(preprocess_features(feat_d))
    feat_a = th.FloatTensor(preprocess_features(feat_a))
    feat_w = th.FloatTensor(preprocess_features(feat_w))

    ma = np.loadtxt(path + 'ma.txt', dtype=int).T
    md = np.loadtxt(path + 'md.txt', dtype=int).T
    mw = np.loadtxt(path + 'mw.txt', dtype=int).T

    # Because none of M, D, A or W has features, we assign one-hot encodings to all of them.
    # mam = sp.load_npz(path + "mam.npz")
    # mdm = sp.load_npz(path + "mdm.npz")
    # mwm = sp.load_npz(path + "mwm.npz")
    pos = sp.load_npz(path + "pos.npz")
    mam, mdm, mwm = generate_metapaths(ma, md, mw)
    pathsim_matrices = pathsim([mam, mdm, mwm])

    # DGL he graph
    hg = dgl.heterograph({
        ('movie', 'ma', 'actor'): (ma[0], ma[1]),
        ('actor', 'am', 'movie'): (ma[1], ma[0]),
        ('movie', 'md', 'direct'): (md[0], md[1]),
        ('direct', 'dm', 'movie'): (md[1], md[0]),
        ('movie', 'mw', 'writer'): (mw[0], mw[1]),
        ('writer', 'wm', 'movie'): (mw[1], mw[0]),
    })
    # hg = dgl.to_bidirected(hg)
    hg.nodes['movie'].data['h'] = feat_m
    hg.nodes['actor'].data['h'] = feat_a
    hg.nodes['direct'].data['h'] = feat_d
    hg.nodes['writer'].data['h'] = feat_w

    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]

    label = th.FloatTensor(label)
    nei_d = [th.LongTensor(i) for i in nei_d]
    nei_a = [th.LongTensor(i) for i in nei_a]
    nei_w = [th.LongTensor(i) for i in nei_w]

    # metapath graph
    mp_g = dgl.heterograph({
        ('movie', 'mam', 'movie'): mam.nonzero(),
        ('movie', 'mdm', 'movie'): mdm.nonzero(),
        ('movie', 'mwm', 'movie'): mwm.nonzero(),
    })

    mam_ = sparse_mx_to_torch_sparse_tensor(normalize_adj(mam))
    mdm_ = sparse_mx_to_torch_sparse_tensor(normalize_adj(mdm))
    mwm_ = sparse_mx_to_torch_sparse_tensor(normalize_adj(mwm))
    consensus_mp = consensus_graph([mam_, mdm_, mwm_])
    pathsim_matrices.append(consensus_mp.to_dense().numpy()*0.5)
    pos = sparse_mx_to_torch_sparse_tensor(pos)
    train = [th.LongTensor(i) for i in train]
    val = [th.LongTensor(i) for i in val]
    test = [th.LongTensor(i) for i in test]

    canonical_etypes = [('movie', 'ma', 'actor'), ('movie', 'md', 'direct'), ('movie', 'mw', 'writer')]
    main_type = 'movie'

    '''生成 train val test'''
    # classes = np.load(path + "labels.npy").astype('int32')
    # ratio = [1, 3, 5, 10]#
    # for i in ratio:
    #     c_train, c_val, c_test = [], [], []
    #     for c in range(3):
    #         c_id = np.where(classes == c)[0]
    #         np.random.shuffle(c_id)
    #         num = int(1000/3)
    #         c_train.append(c_id[:i])
    #         if c == 0 :
    #             c_val.append(c_id[i: num + i + 1])
    #             c_test.append(c_id[num + i + 1: num * 2 + i + 2])
    #         else:
    #             c_val.append(c_id[i: num + i])
    #             c_test.append(c_id[num + i: num * 2 + i])
    #
    #     train = np.concatenate((c_train[0], c_train[1], c_train[2]))
    #     val = np.concatenate((c_val[0], c_val[1], c_val[2]))
    #     test = np.concatenate((c_test[0], c_test[1], c_test[2]))
    #     np.save(path + 'train_' + str(i) + '.npy', train)
    #     np.save(path + 'val_' + str(i) + '.npy', val)
    #     np.save(path + 'test_' + str(i) + '.npy', test)

    if return_hg == True:
        return hg, canonical_etypes, main_type, [feat_m, feat_d, feat_a, feat_w], [mam_, mdm_, mwm_], pathsim_matrices, pos, label, train, val, test
    else:
        return [nei_d, nei_a, nei_w], [feat_m, feat_d, feat_a, feat_w], [mam_, mdm_, mwm_], pos, label, train, val, test

def load_imdb(ratio, type_num, return_hg):
    # The order of node types: 0 m 1 d 2 a 3 w
    M, A, D = 4661, 5841, 2270
    pos_num = 50
    path = "../data/imdb/"
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)

    feat_m = sp.load_npz(path + "m_feat.npz").astype("float32")
    feat_a = sp.load_npz(path + "a_feat.npz").astype("float32")
    feat_d = sp.load_npz(path + "d_feat.npz").astype("float32")

    feat_m = th.FloatTensor(preprocess_features(feat_m))
    feat_a = th.FloatTensor(preprocess_features(feat_a))
    feat_d = th.FloatTensor(preprocess_features(feat_d))

    def generate_metapaths(ma, md):
        ma, md = ma.T, md.T
        ma_ = sp.coo_matrix((np.ones(ma.shape[0]), (ma[:, 0], ma[:, 1])), shape=(M, A)).toarray()
        md_ = sp.coo_matrix((np.ones(md.shape[0]), (md[:, 0], md[:, 1])), shape=(M, D)).toarray()

        mam = np.matmul(ma_, ma_.T) #> 0
        mam = sp.coo_matrix(mam)
        mdm = np.matmul(md_, md_.T) #> 0
        mdm = sp.coo_matrix(mdm)
        return mam, mdm

    def generate_pos(mam, mdm):
        mam = mam / mam.sum(axis=-1).reshape(-1, 1)
        mdm = mdm / mdm.sum(axis=-1).reshape(-1, 1)
        # all = (mam + mdm).A.astype("float32")
        all = (mam + mdm).toarray().astype("float32")

        pos = np.zeros((M, M))
        k = 0
        for i in range(len(all)):
            one = all[i].nonzero()[0]
            if len(one) > pos_num:
                oo = np.argsort(-all[i, one])
                sele = one[oo[:pos_num]]
                pos[i, sele] = 1
                k += 1
            else:
                pos[i, one] = 1
        pos = pos + np.eye(M)
        pos = sp.coo_matrix(pos)
        return pos

    def gen_neibor(edges):
        e_dict = {}
        for i in edges:
            if i[0] not in e_dict:
                e_dict[int(i[0])] = []
                e_dict[int(i[0])].append(int(i[1]))
            else:
                e_dict[int(i[0])].append(int(i[1]))
        e_keys = sorted(e_dict.keys())
        e_nei = [e_dict[i] for i in e_keys]
        e_nei = np.array([np.array(i) for i in e_nei], dtype=int)
        return e_nei

    ma = np.loadtxt(path + 'ma.txt', dtype=int).T
    md = np.loadtxt(path + 'md.txt', dtype=int).T

    # ma = EdgePerturb(ma, 0.05, M, A)
    # md = EdgePerturb(md, 0.05, M, D)

    mam, mdm = generate_metapaths(ma, md)
    pathsim_matrices = pathsim([mam, mdm])
    cal_homophily(mam, label)
    nei_a = gen_neibor(ma.T)
    nei_d = gen_neibor(md.T)

    pos = generate_pos(mam, mdm)
    # DGL he graph
    hg = dgl.heterograph({
        ('movie', 'ma', 'actor'): (ma[0], ma[1]),
        ('actor', 'am', 'movie'): (ma[1], ma[0]),
        ('movie', 'md', 'direct'): (md[0], md[1]),
        ('direct', 'dm', 'movie'): (md[1], md[0])
    })
    # hg = dgl.to_bidirected(hg)
    # hg.nodes['movie'].data['h'] = feat_m
    # hg.nodes['actor'].data['h'] = feat_a
    # hg.nodes['direct'].data['h'] = feat_d

    # metapath graph
    mp_g = dgl.heterograph({
        ('movie', 'mam', 'movie'): mam.nonzero(),
        ('movie', 'mdm', 'movie'): mdm.nonzero(),
        # ('author', 'aps', 'subject'): aps.nonzero(),
    })

    # mp_g.nodes['paper'].data['h'] = feat_p

    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]

    label = th.FloatTensor(label)
    nei_d = [th.LongTensor(i) for i in nei_d]
    nei_a = [th.LongTensor(i) for i in nei_a]

    mam = sparse_mx_to_torch_sparse_tensor(normalize_adj(mam))
    mdm = sparse_mx_to_torch_sparse_tensor(normalize_adj(mdm))
    # mam = sparse_mx_to_torch_sparse_tensor(mam)
    # mdm = sparse_mx_to_torch_sparse_tensor(mdm)
    pos = sparse_mx_to_torch_sparse_tensor(pos)
    train = [th.LongTensor(i) for i in train]
    val = [th.LongTensor(i) for i in val]
    test = [th.LongTensor(i) for i in test]

    canonical_etypes = [('movie', 'ma', 'actor'), ('movie', 'md', 'direct')]
    main_type = 'movie'

    '''生成 train val test'''
    # classes = np.load(path + "labels.npy").astype('int32')
    # ratio = [1, 3, 5, 10]#
    # for i in ratio:
    #     c_train, c_val, c_test = [], [], []
    #     for c in range(3):
    #         c_id = np.where(classes == c)[0]
    #         np.random.shuffle(c_id)
    #         num = int(1000/3)
    #         c_train.append(c_id[:i])
    #         if c == 0 :
    #             c_val.append(c_id[i: num + i + 1])
    #             c_test.append(c_id[num + i + 1: num * 2 + i + 2])
    #         else:
    #             c_val.append(c_id[i: num + i])
    #             c_test.append(c_id[num + i: num * 2 + i])
    #
    #     train = np.concatenate((c_train[0], c_train[1], c_train[2]))
    #     val = np.concatenate((c_val[0], c_val[1], c_val[2]))
    #     test = np.concatenate((c_test[0], c_test[1], c_test[2]))
    #     np.save(path + 'train_' + str(i) + '.npy', train)
    #     np.save(path + 'val_' + str(i) + '.npy', val)
    #     np.save(path + 'test_' + str(i) + '.npy', test)

    # save randomwalk prob

    # sum_mp = mam + mdm
    # sum_mp_index = sum_mp.coalesce().indices()
    # sum_mp = dgl.DGLGraph((sum_mp_index[0], sum_mp_index[1]))
    # prob_matrix = get_connection_prob_matrix(sum_mp, num_walks=10, restart_prob=0.15, walk_length=5)
    # torch.save(prob_matrix, path + 'randomwalk_5' + '.pkl')

    prob_matrix_5 = torch.load(path + 'randomwalk_5' + '.pkl')
    prob_matrix_10 = torch.load(path + 'randomwalk_10' + '.pkl')
    prob_matrix_20 = torch.load(path + 'randomwalk_20' + '.pkl')
    prob_matrix = prob_matrix_5
    if return_hg == True:
        return hg, canonical_etypes, main_type, [feat_m, feat_a, feat_d], [mam, mdm], pathsim_matrices, pos, label, train, val, test
    else:
        return [nei_a, nei_d], [feat_m, feat_a, feat_d], [mam, mdm], pos, label, train, val, test

def load_academic(ratio, type_num, return_hg):
    # The order of node types: 0 m 1 d 2 a 3 w
    A, P, V = 28646, 21044, 18
    pos_num = 50
    path = "../data/academic/"

    het_graph, _ = dgl.load_graphs(
        path + 'graph.bin')
    het_graph = het_graph[0]
    category = 'author'
    meta_path_num = 3  # APA (author-paper-author), APVPA (author-paper-venue-paper-author) and APPA (authorpaper-paper-author)
    ratio = [20, 40, 60]
    ndata = het_graph.ndata
    labels = ndata['label'][category].squeeze()
    N = labels.shape[0]
    for n in het_graph.ntypes:
        n_num = het_graph.number_of_nodes(n)
        print('type:{};num:{}'.format(n, n_num))
    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]

    label = np.load(path + "labels.npy").astype('int32') #+ 1
    label = encode_onehot(label)
    label = th.FloatTensor(label)
    h_dict = {}
    # only dw_embedding
    for n in het_graph.ntypes:
        ndata = het_graph.nodes[n].data
        h_dict[n] = ndata['dw_embedding']
    feat_m = sp.coo_matrix(h_dict[category])
    adj_dict = {e: het_graph.adj(etype=e) for e in het_graph.etypes}

    ap = adj_dict['author-paper'].coo()
    pv = adj_dict['paper-venue'].coo()
    pp = adj_dict['cite'].coo()
    ap_ = sp.coo_matrix((np.ones(ap[0].shape), ap), shape=(A, P))
    pv_ = sp.coo_matrix((np.ones(pv[0].shape), pv), shape=(P, V))
    pp_ = sp.coo_matrix((np.ones(pp[0].shape), pp), shape=(P, P))

    apa = ap_ @ ap_.T

    apv = ap_ @ pv_
    apvpa = apv @ apv.T

    appa = ap_ @ pp_ @ ap_.T

    feat_m = th.FloatTensor(preprocess_features(feat_m))

    def generate_pos(apa, apvpa, appa):
        apa = apa / apa.sum(axis=-1).reshape(-1, 1)
        apvpa = apvpa / apvpa.sum(axis=-1).reshape(-1, 1)
        appa = appa / appa.sum(axis=-1).reshape(-1, 1)
        all = (apa + apvpa + appa).toarray().astype("float32")

        pos = np.zeros((A, A))
        k = 0
        for i in range(len(all)):
            one = all[i].nonzero()[0]
            if len(one) > pos_num:
                oo = np.argsort(-all[i, one])
                sele = one[oo[:pos_num]]
                pos[i, sele] = 1
                k += 1
            else:
                pos[i, one] = 1
        pos = pos + np.eye(A)
        pos = sp.coo_matrix(pos)
        return pos

    pos = ((apa + apvpa + appa).toarray() > 50).astype("float32")
    pos = pos + np.eye(A)
    pos = sp.coo_matrix(pos)

    apa = sparse_mx_to_torch_sparse_tensor(normalize_adj(apa))
    apvpa = sparse_mx_to_torch_sparse_tensor(normalize_adj(apvpa))
    appa = sparse_mx_to_torch_sparse_tensor(normalize_adj(appa))

    pos = sparse_mx_to_torch_sparse_tensor(pos)
    train = [th.LongTensor(i) for i in train]
    val = [th.LongTensor(i) for i in val]
    test = [th.LongTensor(i) for i in test]
    train = [th.LongTensor(i) for i in train]
    val = [th.LongTensor(i) for i in val]
    test = [th.LongTensor(i) for i in test]

    canonical_etypes = [('author', 'author-paper', 'paper'), ('paper', 'cite', 'paper'), ('paper', 'paper-venue', 'venue')]
    main_type = 'author'
    return het_graph, canonical_etypes, main_type, [feat_m, feat_m, feat_m], [apa, apvpa, appa], [apa, apvpa, appa], pos, label, train, val, test

def get_degree(hg, node_cumsum, device, return_dict):
    # 计算target节点的度[n_node, types]
    degree_dict = {}
    node_degree = torch.zeros([hg.num_nodes(), len(hg.ntypes)], device=device)
    for n_type in hg.ntypes:
        n_src = hg.number_of_nodes(n_type)
        index = torch.Tensor(np.arange(n_src)).to(torch.int64).to(device)
        for meta_edge in hg.canonical_etypes:
            src_type, etype, dst_type = meta_edge
            if n_type == src_type:
                src_type_id, dst_type_id = hg.get_ntype_id(src_type), hg.get_ntype_id(dst_type)
                node_degree[index + node_cumsum[src_type_id], dst_type_id] = hg[meta_edge].out_degrees(index, etype=etype).float()
        degree_dict[n_type] = node_degree[index + node_cumsum[hg.get_ntype_id(n_type)]]
    if return_dict == True:
        return degree_dict
    else:
        return node_degree


def encoding_mask_noise(x, mask_rate=0.3, leave_unchanged=0.0, replace_rate=0.0):
    num_nodes = x.shape[0]
    perm = torch.randperm(num_nodes, device=x.device)  # 先获得顺序打乱节点id

    # random masking
    num_mask_nodes = int(mask_rate * num_nodes)  # mask的数量
    mask_nodes = perm[: num_mask_nodes]  # mask的id
    keep_nodes = perm[num_mask_nodes:]  # 保留的id

    perm_mask = torch.randperm(num_mask_nodes, device=x.device)  # 用于后面的mask特征，具体操作（mask or unchange or replace）
    num_leave_nodes = int(leave_unchanged * num_mask_nodes)  # 选择被mask的特征，可能被保留
    num_noise_nodes = int(replace_rate * num_mask_nodes)  # 可能被替换
    num_real_mask_nodes = num_mask_nodes - num_leave_nodes - num_noise_nodes  # 真正被mask的数量
    token_nodes = mask_nodes[perm_mask[: num_real_mask_nodes]]  # 最终的被mask的id
    noise_nodes = mask_nodes[perm_mask[-num_noise_nodes:]]  # 最终的被替换的id
    noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]  # 用来去替换的id

    out_x = x.clone()
    out_x[token_nodes] = 0.0  # 将一部分给mask掉
    # out_x[token_nodes] += self.enc_mask_token  # 给被mask的节点加上 mask token，该token初始化为0，是可学习的
    if num_noise_nodes > 0:
        out_x[noise_nodes] = x[noise_to_be_chosen]  # 将一部分特征 使用其他的特征来替换

    return out_x, (mask_nodes, keep_nodes)

def pathsim_topk(adjs, max_nei):
    print("the number of edges:", [adj.getnnz() for adj in adjs])
    top_adjs = []
    adjs_num = []
    for t in range(len(adjs)):
        A = adjs[t].todense()
        value = []
        x,y = A.nonzero()
        for i,j in zip(x,y):
            value.append(2 * A[i, j] / (A[i, i] + A[j, j]))
        pathsim_matrix = sp.coo_matrix((value, (x, y)), shape=A.shape).toarray()
        idx_x = np.array([np.ones(max_nei[t])*i for i in range(A.shape[0])], dtype=np.int32).flatten()
        idx_y = np.sort(np.argsort(pathsim_matrix, axis=1)[:,::-1][:,0:max_nei[t]]).flatten()
        new = []
        for i,j in zip(idx_x,idx_y):
            new.append(A[i,j])
        new = (np.int32(np.array(new)))
        adj_new = sp.coo_matrix((new, (idx_x,idx_y)), shape=adjs[t].shape)
        adj_num = np.array(new).nonzero()
        adjs_num.append(adj_num[0].shape[0])
        top_adjs.append(adj_new)
    print("the top-k number of edges:", [adj for adj in adjs_num])
    return top_adjs


# def pathsim(A):
#     A = A.todense().astype(int)
#     # 获取图的节点数
#     num_nodes = A.shape[0]
#
#     # 创建一个与邻接矩阵相同大小的相似度矩阵，并初始化为0
#     sim_matrix = np.zeros((num_nodes, num_nodes))
#
#     # 遍历每一对节点(i, j)
#     for i in range(num_nodes):
#         for j in range(i, num_nodes):
#             # 计算两个节点之间的路径个数
#             P_ii = np.dot(A[i, :], A[:, j])
#             P_ij = np.dot(A[i, :], A[:, i])
#             P_jj = np.dot(A[j, :], A[:, j])
#
#             # 使用 PathSim 公式计算相似度
#             if P_ii + P_jj > 0:
#                 sim_matrix[i, j] = (2 * P_ij) / (P_ii + P_jj)
#                 sim_matrix[j, i] = sim_matrix[i, j]  # 对称矩阵
#
#     return sim_matrix

def pathsim(mps):
    pathsim_matrices = []
    I = np.identity(mps[0].shape[0])
    for t in range(len(mps)):
        A = mps[t].todense()
        value = []
        x, y = A.nonzero()
        for i, j in zip(x, y):
            value.append(2 * A[i, j] / (A[i, i] + A[j, j]))
        pathsim_matrix = sp.coo_matrix((value, (x, y)), shape=A.shape).toarray() - I
        pathsim_matrices.append(pathsim_matrix)
    return pathsim_matrices

def mask_node(hg, mask_ratio_dict):
    for n_type in hg.ntypes:
        feat = hg.ndata['h'][n_type]
        out_x, _ = encoding_mask_noise(feat, mask_rate=mask_ratio_dict[n_type], leave_unchanged=0.0,
                                       replace_rate=0.0)
        hg.nodes[n_type].data['h'] = out_x
    return hg

def mask_edge(hg, mask_ratio):
    for meta_edge in hg.canonical_etypes:
        src_type, etype, dst_type = meta_edge
        edges = hg.adj(etype=etype).coalesce().indices().to(hg.device)
        num_edge = edges.size(1)
        index = np.arange(num_edge)
        np.random.shuffle(index)
        mask_num = int(num_edge * mask_ratio)
        mask_index = torch.from_numpy(index[0:mask_num]).type(torch.long).to(hg.device)  # [0:mask_num]
        hg.remove_edges(eids=mask_index, etype=etype)
    return hg

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


def build_topk_graph(features, k):
    """
    根据给定的特征矩阵构建优化的 top-k 图
    :param features: 特征矩阵 (num_nodes, num_features)
    :param k: 每个节点保留的 top-k 邻居
    :return: 邻接矩阵 (num_nodes, num_nodes) (稀疏)
    """
    num_nodes = features.size(0)

    # 计算余弦相似度矩阵 # 归一化特征
    similarity_matrix = torch.mm(F.normalize(features), F.normalize(features).t())  # 余弦相似度矩阵

    # 对角线设置为最小值，防止自己成为top-k邻居
    similarity_matrix.fill_diagonal_(-float('inf'))

    # 获取每个节点的 top-k 最近邻节点
    topk_values, topk_indices = torch.topk(similarity_matrix, k=k, dim=1)

    # 构建邻接矩阵 (稀疏表示)
    rows = torch.arange(num_nodes).unsqueeze(1).expand(num_nodes, k).flatten().to(features.device)
    cols = topk_indices.flatten()
    values = torch.ones(num_nodes * k).to(features.device)

    # 使用稀疏矩阵表示邻接矩阵
    adj_matrix = torch.sparse.FloatTensor(
        torch.stack([rows, cols]), values, torch.Size([num_nodes, num_nodes])
    )

    # 保证图的无向性 (对称)
    adj_matrix = adj_matrix.coalesce()  # 合并重复索引
    adj_matrix = adj_matrix + adj_matrix.t()  # 取对称矩阵
    adj_matrix = adj_matrix.coalesce()  # 再次合并重复索引

    adj_matrix = adj_matrix.to_dense() + torch.eye(num_nodes).to(features.device)
    return adj_matrix


def consensus_graph(mps):
    # 计算交集
    # 初始化交集为第一个稀疏矩阵的稠密格式
    consensus_adj = mps[0].to_dense().bool()

    # 逐个与后续矩阵计算交集
    for matrix in mps[1:]:
        consensus_adj &= matrix.to_dense().bool()

    consensus_adj = consensus_adj.float()
    rowsum = consensus_adj.sum(1)
    # Compute D^(-1/2)
    d_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.  # Handle infinite values
    # Create diagonal matrix D^(-1/2)
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    # Return the normalized adjacency matrix
    adj_normalized = d_mat_inv_sqrt @ consensus_adj @ d_mat_inv_sqrt  # torch.
    return adj_normalized.to_sparse()


def load_data(dataset, ratio, type_num, return_hg):
    if dataset == "acm":
        data = load_acm(ratio, type_num, return_hg)
    elif dataset == "dblp":
        data = load_dblp(ratio, type_num, return_hg)
    elif dataset == "aminer":
        data = load_aminer(ratio, type_num, return_hg)
    elif dataset == "freebase":
        data = load_freebase(ratio, type_num, return_hg)
    elif dataset == "imdb":
        data = load_imdb(ratio, type_num, return_hg)
    elif dataset == 'academic':
        data = load_academic(ratio, type_num, return_hg)
    return data

