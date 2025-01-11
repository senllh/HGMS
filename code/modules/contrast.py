import torch as th
import torch.nn as nn
from GCL.models import DualBranchContrast
import GCL.losses as L
import numpy as np
import torch.nn.functional as F
# from .cluster import *
# import faiss
import torch

def sim(z1, z2):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return th.mm(z1, z2.t())

def homo_loss(x, edge_index, nclusters, niter, sigma):
    kmeans = faiss.Kmeans(x.shape[1], nclusters, niter=niter)
    kmeans.train(x.cpu().detach().numpy())
    ''' kmeans.centroids 是求出质心的表示'''
    centroids = th.FloatTensor(kmeans.centroids).to(x.device)
    logits = []
    for c in centroids:
        '''算出每个节点与每个质心的距离'''
        logits.append((-th.square(x - c).sum(1)/sigma).view(-1, 1))
    logits = th.cat(logits, axis=1)  # [N, C] 每个节点与每个质心的距离
    probs = F.softmax(logits, dim=1)  # softmax用于归一化，使得每个节点的对应不用簇的概率之和为1，也就是每个节点所属每个簇的概率
    loss = F.mse_loss(probs[edge_index[0]], probs[edge_index[1]])  # 缩小相邻节点的聚类差异
    results = torch.argmax(probs, dim=1)
    return loss, probs, results

class Contrast(nn.Module):
    def __init__(self, hidden_dim, tau, lam):
        super(Contrast, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.tau = tau
        self.lam = lam
        for model in self.proj:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)

    def sim(self, z1, z2):
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)
        return sim_matrix

    def forward(self, z1, z2, pos=None, S=None, quantile=0.95):

        if pos == None:
            dense_pos = torch.ones([z1.size(0), z1.size(0)]).to(z1.device)
        else:
            if pos.is_sparse == True:
                dense_pos = pos.to_dense().to(z1.device)
            else:
                dense_pos = pos

        z1_proj = self.proj(z1)
        z2_proj = self.proj(z2)
        matrix_12 = self.sim(z1_proj, z2_proj)
        matrix_21 = matrix_12.t()

        if S == None:
            matrix_12 = matrix_12 / (torch.sum(matrix_12, dim=1).view(-1, 1) + 1e-8)
            lori_1 = -torch.log(matrix_12.mul(dense_pos).sum(dim=-1)).mean()

            matrix_21 = matrix_21 / (torch.sum(matrix_21, dim=1).view(-1, 1) + 1e-8)
            lori_2 = -torch.log(matrix_21.mul(dense_pos).sum(dim=-1)).mean()
        else:
            sampled_S = S[torch.randperm(S.size(0))[:2000]][:, torch.randperm(S.size(0))[:2000]]# 2000
            threshold = torch.quantile(sampled_S, quantile)  # , dim=1
            S = torch.where(S < threshold, S, 1.0)  # .unsqueeze(1)
            S.fill_diagonal_(1)

            pos_12 = (matrix_12 * dense_pos).sum(dim=-1)
            neg_12 = torch.sum(matrix_12 * (1 - dense_pos * S), dim=1)
            lori_1 = -torch.log(pos_12 / (pos_12 + neg_12)).mean()
            #
            pos_21 = (matrix_21 * dense_pos).sum(dim=-1)
            neg_21 = torch.sum(matrix_21 * (1 - dense_pos * S), dim=1)
            lori_2 = -torch.log(pos_21 / (pos_21 + neg_21)).mean()

        return self.lam * lori_1 + (1 - self.lam) * lori_2


class Contrast_S(nn.Module):
    def __init__(self, hidden_dim, tau, lam):
        super(Contrast_S, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.tau = tau
        self.lam = lam
        for model in self.proj:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)

    def sim(self, z1, z2):
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)
        return sim_matrix

    def forward(self, z1, z2, pos=None, S=None):

        if pos == None:
            dense_pos = torch.ones([z1.size(0), z1.size(0)]).to(z1.device)
        else:
            if pos.is_sparse == True:
                dense_pos = pos.to_dense().to(z1.device)
            else:
                dense_pos = pos

        # threshold = torch.quantile(S, 0.95)  # , dim=1
        # S = torch.where(S >= threshold, S, 0.0)  # .unsqueeze(1)
        # S.fill_diagonal_(1)

        z1_proj = self.proj(z1)
        z2_proj = self.proj(z2)
        matrix_12 = self.sim(z1_proj, z2_proj)
        matrix_21 = matrix_12.t()

        # matrix_12 = matrix_12 / (torch.sum(matrix_12, dim=1).view(-1, 1) + 1e-8)
        # lori_1 = -torch.log(matrix_12.mul(dense_pos).sum(dim=-1)).mean()
        #
        # matrix_21 = matrix_21 / (torch.sum(matrix_21, dim=1).view(-1, 1) + 1e-8)
        # lori_2 = -torch.log(matrix_21.mul(dense_pos).sum(dim=-1)).mean()

        # pos_12 = (matrix_12 * dense_pos)
        # neg_12 = (torch.sum(matrix_12, dim=1).view(-1, 1) + 1e-8)
        # lori_1 = -torch.log((pos_12/ neg_12).sum(dim=-1)).mean()
        # #
        # pos_21 = matrix_21 * dense_pos.t()
        # neg_21 = (torch.sum(matrix_21, dim=1).view(-1, 1) + 1e-8)
        # lori_2 = -torch.log((pos_21/ neg_21).sum(dim=-1)).mean()

        pos_12 = (matrix_12 * dense_pos).sum(dim=-1)
        neg_12 = torch.sum(matrix_12 * (1 - dense_pos * S), dim=1)
        lori_1 = -torch.log(pos_12 / (pos_12 + neg_12)).mean()
        #
        pos_21 = (matrix_21 * dense_pos).sum(dim=-1)
        neg_21 = torch.sum(matrix_21 * (1 - dense_pos * S), dim=1)
        lori_2 = -torch.log(pos_21 / (pos_21 + neg_21)).mean()

        return self.lam * lori_1 + (1 - self.lam) * lori_2


class GRAPE(nn.Module):
    def __init__(self, hidden_dim, tau, lam):
        super(GRAPE, self).__init__()
        self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.tau = tau
        self.lam = lam
        # for model in self.proj:
        #     if isinstance(model, nn.Linear):
        #         nn.init.xavier_normal_(model.weight, gain=1.414)

    def projection(self, z):
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def semi_loss(self, z1, adj1, z2, adj2, S, scheme='mask', mean=True):
        def f(x):
            return torch.exp(x / self.tau)

        refl_sim = f(sim(z1, z1))
        between_sim = f(sim(z1, z2))
        if scheme == 'weight':
            ''' pos 直接给定所有正样本
                '''
            if mean:
                positive = between_sim.diag() + (refl_sim * S).sum(1) / (adj1.sum(1) + 0.01)
            else:
                positive = between_sim.diag() + (refl_sim * S).sum(1)
            negative = refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()
        elif scheme == 'mask':
            '''
            pos 就是跨视角的对角线

            neg :对负样本乘一个mask矩阵，使得假负样本的权重很低
            intra-view neg: refl_sim.sum(1) - refl_sim.diag() - (refl_sim*S).sum(1)
            解释: 首先非对角线为负样本，然后再移除一些相似的样本，越是相似移除的比例越高

            inter-view neg: between_sim.sum(1) - (between_sim*S).sum(1)
            解释：所有样本移除相似的样本，越是相似移除的比例越高

            疑问: 为什么 inter-view neg 中不包括 between_sim.diag()

            讨论pos + neg
            intra-view: refl_sim.sum(1) - refl_sim.diag() - (refl_sim*S).sum(1)
            解释: 在这个视角中缺少了 intra的正样本信息， 所以 pos+neg != refl_sim.sum(1)

            inter-view：between_sim.sum(1) - (between_sim*S).sum(1) + between_sim.diag()
            解释: pos+neg != between_sim.sum(1),也缺少了一些信息，某些对角线的信息还会多
            '''
            positive = between_sim.diag()
            negative = refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag() - (
                    (between_sim * S).sum(1) + (refl_sim * S).sum(1))

        loss = -torch.log(positive / (positive + negative))

        return loss

    def forward(self, z1, graph1, z2, graph2, S, scheme='mask', mean=True):

        h1 = self.projection(z1)
        h2 = self.projection(z2)

        # h1 = z1
        # h2 = z2

        l1 = self.semi_loss(h1, graph1, h2, graph2, S, scheme, mean)
        l2 = self.semi_loss(h2, graph2, h1, graph1, S, scheme, mean)

        ret = (l1 + l2) * 0.5
        ret = ret.mean()

        return ret


class HomoGCL(nn.Module):
    def __init__(self, hidden_dim, tau):
        super(HomoGCL, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.tau = tau
        # self.lam = lam
        for model in self.proj:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)

    def inter_loss(self, z1, z2):
        f = lambda x: th.exp(x / self.tau)  # 标准的infoNCE
        between_sim = f(sim(z1, z2))  # inter 相似度
        pos = between_sim.diag()
        neg = between_sim.sum(1) - between_sim.diag()
        loss = -th.log(pos / (pos + neg))
        return loss

    def inter_loss_neighbor(self, z1, adj1, z2, adj2, confmatrix):
        f = lambda x: th.exp(x / self.tau)  # 标准的infoNCE
        between_sim = f(sim(z1, z2))  # inter 相似度
        pos = between_sim.diag() #+ (between_sim * adj1).sum(1) / (adj1.sum(1)+0.01)
        neg = between_sim.sum(1) - between_sim.diag() - (between_sim * adj1).sum(1)
        loss = -th.log(pos / (pos + neg))
        return loss

    def intra_loss(self, z1):
        f = lambda x: th.exp(x / self.tau)  # 标准的infoNCE
        refl_sim = f(sim(z1, z1))  # intra 相似度
        pos = refl_sim.diag()
        neg = refl_sim.sum(1) - refl_sim.diag()
        loss = -th.log(pos / (pos + neg))
        return loss

    def intra_loss_neighbor(self, z, adj):
        f = lambda x: th.exp(x / self.tau)  # 标准的infoNCE
        refl_sim = f(sim(z, z))  # intra 相似度
        pos = refl_sim.diag() + (refl_sim * adj).sum(1) / (adj.sum(1)+0.01)
        neg = refl_sim.sum(1) - refl_sim.diag() - (refl_sim * adj).sum(1)
        # loss = -th.log(pos / (pos + neg))
        loss = -th.log(pos / (neg))
        return loss

    def our_loss_pos(self, z1, adj1, z2, adj2, confmatrix):
        f = lambda x: th.exp(x / self.tau)  # 标准的infoNCE
        refl_sim = f(sim(z1, z1))  # intra 相似度
        between_sim = f(sim(z1, z2))  # inter 相似度
        pos = between_sim.diag() + (refl_sim * adj1 * confmatrix).sum(1) / (adj1.sum(1) + 0.01) #+ \
              #(between_sim * adj1 * confmatrix).sum(1) / (adj1.sum(1) + 0.01)
        neg = refl_sim.sum(1) - refl_sim.diag() + between_sim.sum(1)
        loss = -th.log(pos / (pos + neg))
        return loss

    def our_loss_mask(self, z1, adj1, z2, adj2, confmatrix):
        f = lambda x: th.exp(x / self.tau)  # 标准的infoNCE
        refl_sim = f(sim(z1, z1))  # intra 相似度
        between_sim = f(sim(z1, z2))  # inter 相似度
        pos = between_sim.diag()
        neg = refl_sim.sum(1) - (refl_sim * adj1).sum(1) \
              + between_sim.sum(1) - (between_sim * adj1).sum(1)
        loss = -th.log(pos / (pos + neg))
        return loss

    def semi_loss(self, z1, adj1, z2, adj2, confmatrix, mean):
        f = lambda x: th.exp(x / self.tau)  # 标准的infoNCE
        refl_sim = f(sim(z1, z1))     # intra 相似度
        between_sim = f(sim(z1, z2))  # inter 相似度
        ''' between_sim.diag() 为 两种视角的 instance-wise的正样本对
            refl_sim * adj1 是使得同一个视角中 相连节点为正样本
            * confmatrix 则乘了（两个节点之间的聚类相似度），一种软的放缩'''
        if mean:
            pos = between_sim.diag() + (refl_sim * adj1 * confmatrix).sum(1) / (adj1.sum(1)+0.01)
                  # + \
                  # (between_sim * adj2 * confmatrix).sum(1) / (adj2.sum(1)+0.01)
            # pos = between_sim.diag()
        else:
            pos = between_sim.diag() + (refl_sim * adj1 * confmatrix).sum(1)
        '''负样本的构成：
        intra-view 的负样本：refl_sim - refl_sim.diag() - (refl_sim * adj1).sum(1)
        也就是排除 对角线 和相连的节点 其他都为负样本
        inter-view 的负样本：between_sim.sum(1) - (between_sim * adj2).sum(1)
        '''
        neg = refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag() - (refl_sim * adj1).sum(1) - (between_sim * adj2).sum(1)
        # neg = between_sim.diag() - refl_sim
        loss = -th.log(pos / (pos + neg))
        return loss

    def forward(self, z1, graph1, z2, graph2, confmatrix, mean):
        h1 = self.proj(z1)
        h2 = self.proj(z2)

        # h1 = z1
        # h2 = z2

        # l1 = self.semi_loss(h1, graph1, h2, graph2, confmatrix, mean)
        # l2 = self.semi_loss(h2, graph2, h1, graph1, confmatrix, mean)
        #
        # l1 = self.inter_loss(h1, h2)
        # l2 = self.inter_loss(h2, h1)
        #
        # l1 = self.inter_loss_neighbor(h1, graph1, h2, graph2, confmatrix)
        # l2 = self.inter_loss_neighbor(h2, graph2, h1, graph1, confmatrix)

        # l1 = self.intra_loss_neighbor(h1, graph1) + self.inter_loss(h1, h2)
        # l2 = self.intra_loss_neighbor(h2, graph2) + self.inter_loss(h1, h2)

        l1 = self.our_loss_pos(h1, graph1, h2, graph2, confmatrix)
        l2 = self.our_loss_pos(h2, graph2, h1, graph1, confmatrix)

        # l1 = self.our_loss_mask(h1, graph1, h2, graph2, confmatrix)
        # l2 = self.our_loss_mask(h2, graph2, h1, graph1, confmatrix)

        ret = (l1 + l2) * 0.5
        ret = ret.mean()

        return ret


class Proto_Contrast(nn.Module):
    def __init__(self, hidden_dim, tau, lam):
        super(Proto_Contrast, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.tau = tau
        self.lam = lam
        for model in self.proj:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)

    def sim(self, z1, z2):
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)
        return sim_matrix

    def forward(self, z1, z2, num_clusters):

        cluster_result = {'im2cluster': [], 'centroids': [], 'density': []}
        # for num_cluster in num_clusters:
        #     cluster_result['im2cluster'].append(torch.zeros(z1.size(0), dtype=torch.long).cuda())
        #     cluster_result['centroids'].append(torch.zeros(num_cluster, z1.size(1)).cuda())
        #     cluster_result['density'].append(torch.zeros(num_cluster).cuda())
        cluster_result['im2cluster'].append(torch.zeros(z1.size(0), dtype=torch.long).cuda())
        cluster_result['centroids'].append(torch.zeros(num_clusters, z1.size(1)).cuda())
        cluster_result['density'].append(torch.zeros(num_clusters).cuda())
        cluster_result = run_kmeans(z1+z2, [num_clusters], 10)

        N = z1.size(0)
        node_idx = torch.range(0, N - 1).long().cuda()
        cluster_idx = cluster_result['im2cluster'][0]#[i]
        idx = torch.stack((node_idx, cluster_idx), 0)
        data = torch.ones(N).cuda()
        coo_i = torch.sparse_coo_tensor(idx, data, [N, num_clusters])  # 将节点和簇之间的所属关系构建为一个稀疏矩阵
        dense_pos = torch.mm(coo_i, coo_i.to_dense().t())
        dense_pos = torch.eye(dense_pos.size(0)).cuda()
        # print(coo_i.coalesce().indices()[1, :].unique())
        z1_proj = self.proj(z1)
        z2_proj = self.proj(z2)
        matrix_12 = self.sim(z1_proj, z2_proj)
        matrix_21 = matrix_12.t()

        # matrix_12 = matrix_12/(torch.sum(matrix_12, dim=1).view(-1, 1) + 1e-8)
        # lori_1 = -torch.log(matrix_12.mul(dense_pos).sum(dim=-1)).mean()
        #
        # matrix_21 = matrix_21 / (torch.sum(matrix_21, dim=1).view(-1, 1) + 1e-8)
        # lori_2 = -torch.log(matrix_21.mul(dense_pos).sum(dim=-1)).mean()
        #

        matrix_12 = matrix_12.diag() / matrix_12.sum(1)  # Pos / Pos + Neg
        lori_1 = -torch.log(matrix_12).mean()
        matrix_21 = matrix_21.diag() / matrix_21.sum(1)  # Pos / Pos + Neg
        lori_2 = -torch.log(matrix_21).mean()


        # matrix_12 = matrix_12.diag() / (matrix_12.sum(1) - matrix_12.diag())  # Pos/Neg
        # lori_1 = -torch.log(matrix_12).mean()
        # matrix_21 = matrix_21.diag() / (matrix_21.sum(1) - matrix_21.diag())  # Pos/Neg
        # lori_2 = -torch.log(matrix_21).mean()

        return self.lam * lori_1 + (1 - self.lam) * lori_2


class Local_Global_Contrast(nn.Module):
    def __init__(self, hidden_dim, tau, lam):
        super(Local_Global_Contrast, self).__init__()
        self.local_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.global_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.tau = tau
        self.lam = lam
        for model in self.local_proj:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)

        for model in self.global_proj:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)

        self.contrast_model = DualBranchContrast(loss=L.InfoNCE(0.2), mode='G2L').cuda()
    def sim(self, z1, z2):
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)
        return sim_matrix

    def forward(self, z1, z2):

        c1, c2 = z1.mean(dim=0), z2.mean(dim=0)
        c1, c2 = c1.unsqueeze(0), c2.unsqueeze(0)

        z_proj_1 = self.local_proj(z1)
        z_proj_2 = self.local_proj(z2)

        idx1 = np.random.permutation(z1.size(0))
        idx2 = np.random.permutation(z1.size(0))
        h3 = z_proj_1[idx1]
        h4 = z_proj_2[idx2]

        c_proj_1 = self.global_proj(c1)
        c_proj_2 = self.global_proj(c2)

        return self.contrast_model(h1=z_proj_1, h2=z_proj_2, g1=c_proj_1, g2=c_proj_2, h3=h3, h4=h4)

class IntraContrast(nn.Module):
    def __init__(self, hidden_dim, tau, lam):
        super(IntraContrast, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.tau = tau
        self.lam = lam
        for model in self.proj:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)

    def sim(self, z1, z2):
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)
        return sim_matrix

    def forward(self, z, pos):
        dense_pos = pos.to_dense()

        z_proj = self.proj(z)
        matrix_sim = self.sim(z_proj, z_proj)

        matrix_sim = matrix_sim / (torch.sum(matrix_sim, dim=1).view(-1, 1) + 1e-8)
        lori_mp = -torch.log(matrix_sim.mul(dense_pos).sum(dim=-1)).mean()
        return lori_mp


class Info_and_Proto(nn.Module):
    def __init__(self, hidden_dim, tau, lam):
        super(Info_and_Proto, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.tau = tau
        self.lam = lam
        for model in self.proj:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)

    def sim(self, z1, z2):
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)
        return sim_matrix

    def ProtoNCE(self, z, cluster_result):
        # z 是粗粒度的表示
        loss = torch.tensor(0.0, requires_grad=True).cuda()
        for _, (im2cluster, prototypes, density) in enumerate(zip(cluster_result['im2cluster'], \
                                                                  cluster_result['centroids'],
                                                                  cluster_result['density'])):
            # 遍历聚类的results, prototypes 是质心的表示
            node_prototypes = prototypes[im2cluster]  # 以此获得每个节点对应质心的表示
            phi = density[im2cluster]  # 获得每个节点对应质心的密度
            pos_prototypes = torch.exp(torch.mul(z, node_prototypes).sum(axis=1) / phi)  # 正样本，每个节点都应该和其对应的簇表示拉近距离
            neg_prototypes = torch.exp(torch.mm(z, prototypes.t()) / density).mean(axis=1) * z.size(0)  # 每个节点和所有簇的表示相似度
            loss = loss + ((-1) * (torch.log(pos_prototypes / neg_prototypes))).mean()
        loss = loss / len(cluster_result['im2cluster'])
        return loss

    def Clust_InfoNCE(self, z1, z2, cluster_result, num_clusters):
        # z_tar 粗粒度; z 细粒度
        # (z×z.T)/tau
        dots = torch.exp(torch.mm(z1, z2.t()) / self.tau)
        z_min = torch.diag(dots)  # torch.diag 提取对角线元素，也就是 node-node 为正样本 （instance-wise 对比）

        im2cluster = cluster_result['im2cluster']
        im2cluster = torch.stack(im2cluster)
        k_times = len(num_clusters)  # 聚类次数
        N = z1.size(0)  # number of nodes

        weight = torch.ones([N, N]).cuda() * k_times
        for i in range(k_times):
            node_idx = torch.range(0, N - 1).long().cuda()
            cluster_idx = im2cluster[i]
            idx = torch.stack((node_idx, cluster_idx), 0)
            data = torch.ones(N).cuda()
            coo_i = torch.sparse_coo_tensor(idx, data, [N, num_clusters[i]])  # 将节点和簇之间的所属关系构建为一个稀疏矩阵
            weight = weight - torch.mm(coo_i, coo_i.to_dense().t())
            # 上面公式中, 第二项 通过矩阵乘积来获得属于同一簇的节点对
        dots = torch.mul(dots, weight)  # 相似度矩阵 元素乘积 weight; weight（两个节点越属于同一个簇，那么将其移除负样本集合）

        nominator = z_min  # 正样本
        denominator = dots.mean(axis=1) * N  # 正样本+负样本 （经过上面操作，负样本集合中移除过同簇的节点）
        loss = ((-1) * (torch.log(nominator / denominator))).mean()
        return loss

    def forward(self, z1, z2, num_clusters):

        z1_proj = self.proj(z1)
        z2_proj = self.proj(z2)
        z1 = F.normalize(z1_proj, dim=1)#torch.norm(z1_proj, dim=-1, keepdim=True)
        z2 = F.normalize(z2_proj, dim=1)#torch.norm(z2_proj, dim=-1, keepdim=True)

        '''z_anchor: 粗粒度, z: 细粒度'''
        cluster_result = {'im2cluster': [], 'centroids': [], 'density': []}
        for num_cluster in num_clusters:
            cluster_result['im2cluster'].append(torch.zeros(z1.size(0), dtype=torch.long).cuda())
            cluster_result['centroids'].append(torch.zeros(num_cluster, z1.size(1)).cuda())
            cluster_result['density'].append(torch.zeros(num_cluster).cuda())
        cluster_result = run_kmeans(z1, num_clusters, self.tau)

        # z1_proj = self.proj(z1)
        # z2_proj = self.proj(z2)
        '''F.normalize(x) = x / torch.norm(x)
        F.normalize()用于对张量进行归一化处理，使其在某个维度上的范数为1。它主要用于归一化操作，比如将向量的模长归一化。
        torch.norm() 用于计算张量的范数。可以计算不同类型的范数，如L1范数、L2范数等。
        '''
        # z1 = F.normalize(z1_proj, dim=1)#torch.norm(z1_proj, dim=-1, keepdim=True)
        # z2 = F.normalize(z2_proj, dim=1)#torch.norm(z2_proj, dim=-1, keepdim=True)

        loss_info = self.Clust_InfoNCE(z1, z2, cluster_result, num_clusters)
        loss_proto = self.ProtoNCE(z1, cluster_result)

        return self.lam * loss_info + (1 - self.lam) * loss_proto