import dgl
import numpy
import numpy as np
import torch
from utils.evaluate import evaluate, evaluate_cluster
from utils.load_data_DGL import load_data, build_topk_graph
from utils.params import set_params
from modules.model import HGMS
import warnings
import datetime
import random
from copy import deepcopy

warnings.filterwarnings('ignore')
args = set_params()
if torch.cuda.is_available():
    device = torch.device("cuda:" + str(args.gpu))
    torch.cuda.set_device(args.gpu)
else:
    device = torch.device("cpu")

args.device = device
## name of intermediate document ##
own_str = args.dataset

## random seed ##
seed = args.seed
numpy.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


def dense_to_sparse_x(feat_index, n_node, n_dim):
    return torch.sparse.FloatTensor(feat_index, torch.ones(feat_index.shape[1]).to(feat_index.device),
                                    [n_node, n_dim])


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

def filter_noise(mp, y):
    y = y.argmax(1)
    if mp.is_sparse == True:
        edges, weight = mp.coalesce().indices(), mp.coalesce().values()
        src, dst = edges
        mask_id = y[src] == y[dst]
        src, dst = src[mask_id], dst[mask_id]
        weight = weight[mask_id]
        return torch.sparse.FloatTensor(torch.stack([src, dst]), weight, mp.shape)
    else:
        edges = mp.nonzero().T
        src, dst = edges[0], edges[1]
        mask_id = y[src] == y[dst]
        src, dst = src[mask_id], dst[mask_id]
        return torch.sparse.FloatTensor(torch.stack([src, dst]), torch.ones([src.size(0)]).to(device), mp.shape)

def train():
    If_pre_training = True#True, False
    print(args)
    hg, canonical_etypes, main_type, feats, mps, pathsims, pos, label, idx_train, idx_val, idx_test = \
        load_data(args.dataset, args.ratio, args.type_num, return_hg=True)
    nb_classes = label.shape[-1]  # 节点类型的个数
    feats_dim_list = [i.shape[1] for i in feats]  # 不同类型节点的维度
    N = feats[0].size(0)
    P = int(len(mps))  # 元路径的个数

    print("seed ", args.seed)
    print("Dataset: ", args.dataset)
    print("The number of meta-paths: ", P)

    # HeCo是预训练任务
    model = HGMS(args, feats_dim_list, P)

    if torch.cuda.is_available():
        print('Using CUDA')
        model.to(device)
        hg = hg.to(device)
        feats = [feat.to(device) for feat in feats]
        ori_g = dgl.to_homogeneous(hg).adj().cuda()
        mps = [mp.to(device) for mp in mps]
        pathsims = [torch.Tensor(mp).to(device) for mp in pathsims]
        pos = pos.to(device)
        label = label.to(device)
        idx_train = [i.to(device) for i in idx_train]
        idx_val = [i.to(device) for i in idx_val]
        idx_test = [i.to(device) for i in idx_test]


    cnt_wait = 0
    best = 1e9
    best_t = 0
    starttime = datetime.datetime.now()
    topk_graph = build_topk_graph(feats[0], args.topk)
    refine_mps = mps
    S = feats[0] @ feats[0].T
    if If_pre_training == True:
        optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_coef)
        for epoch in range(args.nb_epochs):
            epoch_loss = 0.0
            model.train()
            optimiser.zero_grad()

            x = feats[0]
            (con_loss, self_loss), S, refine_mps = model(args, hg, ori_g, x, refine_mps, S, topk_graph, pathsims, pos, canonical_etypes, epoch)
            loss = con_loss + self_loss * args.sigma
            loss.backward()
            torch.cuda.empty_cache()
            optimiser.step()
            epoch_loss += loss

            print('Epoch--', epoch, "con loss ", con_loss, 'selfexpress loss', self_loss)
            if epoch_loss < best:
                best = epoch_loss
                best_t = epoch
                cnt_wait = 0
                best_model = deepcopy(model.state_dict())
                best_S = deepcopy(S.detach())
            else:
                cnt_wait += 1

            if cnt_wait == args.patience:
                torch.save(best_model, 'strategy3_' + own_str + '.pkl')
                torch.save(best_S, 'strategy3_' + own_str  + '_' + args.self_expressive + 'S.pt')
                print('Early stopping!')
                break
    '''预训练完成'''
    print('Loading {}th epoch'.format(best_t))
    model.load_state_dict(torch.load('strategy3_' + own_str + '.pkl'))
    model.eval()
    embeds, zs = model.get_embeds(feats[0], mps, fc=True)

    '''节点分类任务'''
    print('Node classification task.')
    for i in range(len(idx_train)):
        evaluate(embeds, args.ratio[i], idx_train[i], idx_val[i], idx_test[i], label, nb_classes, device, args.dataset,
                 args.eva_lr, args.eva_wd)

    # '''节点聚类'''
    print('Node clustering task.')
    nmi_list, ari_list = [], []
    embeds = embeds.cpu().data.numpy()
    label = np.argmax(label.cpu().data.numpy(), axis=-1)
    for kmeans_random_state in range(10):
        nmi, ari = evaluate_cluster(embeds, label, nb_classes, kmeans_random_state)
        nmi_list.append(nmi)
        ari_list.append(ari)
    print("\t[clustering] nmi: [{:.4f}, {:.4f}] ari: [{:.4f}, {:.4f}]".format(np.mean(nmi_list), np.std(nmi_list),
                                                                              np.mean(ari_list), np.std(ari_list)))

    '''聚类可视化'''
    # show_cluster_vis(embeds, label, args.dataset, nb_classes)

if __name__ == '__main__':
    train()
