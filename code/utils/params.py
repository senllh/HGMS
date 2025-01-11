import argparse
import sys

argv = sys.argv
dataset = 'acm'#'aminer'#'acm' freebase, dblp, imdb, yelp, academic
self_express = 'network'  # closed network

def acm_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--self_expressive', type=str, default='network')
    parser.add_argument('--save_emb', action="store_true")
    parser.add_argument('--turn', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="acm")
    parser.add_argument('--ratio', type=int, default=[20, 40, 60])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--hidden_dim', type=int, default=256) #64聚类
    parser.add_argument('--nb_epochs', type=int, default=10000)#10000

    # The parameters of evaluation
    parser.add_argument('--eva_lr', type=float, default=0.05)
    parser.add_argument('--eva_wd', type=float, default=0)

    # The parameters of learning process
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001) # 0.001
    parser.add_argument('--l2_coef', type=float, default=1e-6)#1e-6
    parser.add_argument('--use_pos', type=bool, default=False)  # mp 增广用 True，hg 增广用 False
    parser.add_argument('--use_normalize', type=bool, default=False)  # mp 增广用 True，hg 增广用 False

    # model-specific parameters
    parser.add_argument('--tau', type=float, default=0.8) # 0.8
    parser.add_argument('--feat_drop', type=float, default=0.0)
    parser.add_argument('--attn_drop', type=float, default=0.5) # 0.5
    parser.add_argument('--lam', type=float, default=0.5)
    parser.add_argument('--activate', type=str, default='leaky_relu')
    parser.add_argument('--MLP_layers', type=int, default=1)
    # 数据增广
    parser.add_argument('--feat_mask_ratio1', type=float, default=0.5)
    parser.add_argument('--edge_mask_ratio1', type=float, default=0.6)#0.0
    parser.add_argument('--feat_mask_ratio2', type=float, default=0.5)
    parser.add_argument('--edge_mask_ratio2', type=float, default=0.6)

    # muti-view self-express network
    parser.add_argument('--lmbd', type=float, default=0.9)
    parser.add_argument('--express_layer', type=int, default=2)
    parser.add_argument('--express_l2', type=int, default=1e-2)
    parser.add_argument('--beta1', type=int, default=0.1)# 1e-2
    parser.add_argument('--beta2', type=int, default=0.1)# 1e-2

    # closed muti-view self-express
    parser.add_argument('--topk', type=int, default=4)  # 0.1 73.12 78.78
    parser.add_argument('--alpha1', type=float, default=0.000)
    parser.add_argument('--alpha2', type=float, default=0.00)
    parser.add_argument('--beta', type=float, default=1)

    parser.add_argument('--quantile1', type=float, default=0.9)
    parser.add_argument('--quantile2', type=float, default=0.8)
    parser.add_argument('--interval', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.2)
    parser.add_argument('--sigma', type=float, default=0.5)#0.8
    args, _ = parser.parse_known_args()
    args.type_num = [4019, 7167, 60]  # the number of every node type
    args.nei_num = 2  # the number of neighbors' types
    return args

def acm_closed_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--self_expressive', type=str, default='closed')
    parser.add_argument('--save_emb', action="store_true")
    parser.add_argument('--turn', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="acm")
    parser.add_argument('--ratio', type=int, default=[20, 40, 60])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--hidden_dim', type=int, default=256) #64聚类
    parser.add_argument('--nb_epochs', type=int, default=10000)#10000

    # The parameters of evaluation
    parser.add_argument('--eva_lr', type=float, default=0.05)
    parser.add_argument('--eva_wd', type=float, default=0)

    # The parameters of learning process
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001) # 0.001
    parser.add_argument('--l2_coef', type=float, default=1e-6)#1e-6
    parser.add_argument('--use_pos', type=bool, default=False)  # mp 增广用 True，hg 增广用 False
    parser.add_argument('--use_normalize', type=bool, default=False)  # mp 增广用 True，hg 增广用 False

    # model-specific parameters
    parser.add_argument('--tau', type=float, default=0.8) # 0.8
    parser.add_argument('--feat_drop', type=float, default=0.0)
    parser.add_argument('--attn_drop', type=float, default=0.5) # 0.5
    parser.add_argument('--lam', type=float, default=0.5)
    parser.add_argument('--activate', type=str, default='leaky_relu')
    parser.add_argument('--MLP_layers', type=int, default=1)
    # parser.add_argument('--gnn_layer', type=int, default=3)
    # parser.add_argument('--head', type=int, default=8)
    # 数据增广
    parser.add_argument('--feat_mask_ratio1', type=float, default=0.5)
    parser.add_argument('--edge_mask_ratio1', type=float, default=0.6)#0.0
    parser.add_argument('--feat_mask_ratio2', type=float, default=0.5)
    parser.add_argument('--edge_mask_ratio2', type=float, default=0.6)

    # muti-view self-express network
    parser.add_argument('--lmbd', type=float, default=0.9)
    parser.add_argument('--express_layer', type=int, default=2)
    parser.add_argument('--express_l2', type=int, default=1e-2)
    parser.add_argument('--beta1', type=int, default=0.1)# 1e-2
    parser.add_argument('--beta2', type=int, default=0.1)# 1e-2

    # closed muti-view self-express
    parser.add_argument('--outlier', type=float, default=0.999)#
    parser.add_argument('--topk', type=int, default=4)  # 0.1 73.12 78.78
    parser.add_argument('--alpha1', type=float, default=0.000)
    parser.add_argument('--alpha2', type=float, default=0.00)
    parser.add_argument('--beta', type=float, default=1)

    parser.add_argument('--quantile1', type=float, default=0.9)
    parser.add_argument('--quantile2', type=float, default=0.5)
    parser.add_argument('--interval', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.2)
    parser.add_argument('--sigma', type=float, default=0.2)#0.8
    args, _ = parser.parse_known_args()
    args.type_num = [4019, 7167, 60]  # the number of every node type
    args.nei_num = 2  # the number of neighbors' types
    return args

def dblp_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--self_expressive', type=str, default='network')
    parser.add_argument('--save_emb', action="store_true")
    parser.add_argument('--turn', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="dblp")
    parser.add_argument('--ratio', type=int, default=[20, 40, 60])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=53)#53
    parser.add_argument('--hidden_dim', type=int, default=1024)# 1024
    parser.add_argument('--nb_epochs', type=int, default=10000)
    parser.add_argument('--warmup', type=int, default=0)
    # The parameters of evaluation
    parser.add_argument('--eva_lr', type=float, default=0.01)
    parser.add_argument('--eva_wd', type=float, default=0)

    # The parameters of learning process
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.0008)#0.0008
    parser.add_argument('--l2_coef', type=float, default=0)
    parser.add_argument('--use_pos', type=bool, default=True)  # mp 增广用 True，hg 增广用 False
    parser.add_argument('--use_normalize', type=bool, default=True)  # mp 增广用 True，hg 增广用 False


    # model-specific parameters
    parser.add_argument('--tau', type=float, default=0.9)#0.9
    parser.add_argument('--feat_drop', type=float, default=0.4)#0.4
    parser.add_argument('--attn_drop', type=float, default=0.35)#0.35
    parser.add_argument('--lam', type=float, default=0.5)
    parser.add_argument('--activate', type=str, default='leaky_relu')
    parser.add_argument('--MLP_layers', type=int, default=1)
    parser.add_argument('--feat_aug', type=str, default='drop')# 'add' or 'drop'

    # 数据增广
    parser.add_argument('--feat_mask_ratio1', type=float, default=0.0)#0.5
    parser.add_argument('--edge_mask_ratio1', type=float, default=0.3)#0.0
    parser.add_argument('--feat_mask_ratio2', type=float, default=0.0)#0.5
    parser.add_argument('--edge_mask_ratio2', type=float, default=0.3)

    # muti-view self-express network
    parser.add_argument('--lmbd', type=float, default=0.9)
    parser.add_argument('--express_layer', type=int, default=2)
    parser.add_argument('--express_l2', type=int, default=1e-2)
    parser.add_argument('--beta1', type=int, default=0.1)# 1e-2
    parser.add_argument('--beta2', type=int, default=0.1)# 1e-2
    # closed muti-view self-express
    parser.add_argument('--topk', type=int, default=5)  # 0.1 73.12 78.78
    parser.add_argument('--alpha1', type=float, default=0.1) # 0.1 73.12 78.78
    parser.add_argument('--alpha2', type=float, default=0.00) #
    parser.add_argument('--beta', type=float, default=20) # 20 71.73 77

    # filter S
    parser.add_argument('--quantile1', type=float, default=0.8)
    parser.add_argument('--quantile2', type=float, default=0.95)#0.98 0.99
    parser.add_argument('--interval', type=float, default=0)
    parser.add_argument('--gamma', type=float, default=0.2)
    parser.add_argument('--sigma', type=float, default=0.2)#0.8
    args, _ = parser.parse_known_args()
    args.type_num = [4057, 14328, 7723, 20]  # the number of every node type
    return args

def dblp_closed_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--self_expressive', type=str, default='closed')
    parser.add_argument('--save_emb', action="store_true")
    parser.add_argument('--turn', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="dblp")
    parser.add_argument('--ratio', type=int, default=[20, 40, 60])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=53)#53
    parser.add_argument('--hidden_dim', type=int, default=1024)# 1024
    parser.add_argument('--nb_epochs', type=int, default=10000)
    parser.add_argument('--warmup', type=int, default=0)
    # The parameters of evaluation
    parser.add_argument('--eva_lr', type=float, default=0.01)
    parser.add_argument('--eva_wd', type=float, default=0)

    # The parameters of learning process
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.0008)#0.0008
    parser.add_argument('--l2_coef', type=float, default=0)
    parser.add_argument('--use_pos', type=bool, default=True)  # mp 增广用 True，hg 增广用 False
    parser.add_argument('--use_normalize', type=bool, default=True)  # mp 增广用 True，hg 增广用 False


    # model-specific parameters
    parser.add_argument('--tau', type=float, default=0.9)#0.9
    parser.add_argument('--feat_drop', type=float, default=0.4)#0.4
    parser.add_argument('--attn_drop', type=float, default=0.35)#0.35
    parser.add_argument('--lam', type=float, default=0.5)
    parser.add_argument('--activate', type=str, default='leaky_relu')
    parser.add_argument('--MLP_layers', type=int, default=1)
    parser.add_argument('--feat_aug', type=str, default='drop')# 'add' or 'drop'

    # 数据增广
    parser.add_argument('--feat_mask_ratio1', type=float, default=0.0)#0.5
    parser.add_argument('--edge_mask_ratio1', type=float, default=0.3)#0.0
    parser.add_argument('--feat_mask_ratio2', type=float, default=0.0)#0.5
    parser.add_argument('--edge_mask_ratio2', type=float, default=0.3)

    # muti-view self-express network
    parser.add_argument('--lmbd', type=float, default=0.9)
    parser.add_argument('--express_layer', type=int, default=2)
    parser.add_argument('--express_l2', type=int, default=1e-2)
    parser.add_argument('--beta1', type=int, default=0.1)# 1e-2
    parser.add_argument('--beta2', type=int, default=0.1)# 1e-2
    # closed muti-view self-express
    parser.add_argument('--topk', type=int, default=5)  # 0.1 73.12 78.78
    parser.add_argument('--alpha1', type=float, default=0.1) # 0.1 73.12 78.78
    parser.add_argument('--alpha2', type=float, default=0.01) #
    parser.add_argument('--beta', type=float, default=20) # 20

    # filter S
    parser.add_argument('--outlier', type=float, default=0.999)#
    parser.add_argument('--quantile1', type=float, default=0.75)
    parser.add_argument('--quantile2', type=float, default=0.95)#0.98 0.99
    parser.add_argument('--interval', type=float, default=0)
    parser.add_argument('--gamma', type=float, default=0.2)
    parser.add_argument('--sigma', type=float, default=0.2)#0.8
    args, _ = parser.parse_known_args()
    args.type_num = [4057, 14328, 7723, 20]  # the number of every node type
    return args


def aminer_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--self_expressive', type=str, default='network')  # network, closed

    parser.add_argument('--save_emb', action="store_true")
    parser.add_argument('--turn', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="aminer")
    parser.add_argument('--ratio', type=int, default=[20, 40, 60])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)#4, 0,
    parser.add_argument('--hidden_dim', type=int, default=256)#256
    parser.add_argument('--nb_epochs', type=int, default=1000)# 10000
    # The parameters of evaluation
    parser.add_argument('--eva_lr', type=float, default=0.01)
    parser.add_argument('--eva_wd', type=float, default=0)

    # The parameters of learning process
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001) # 0.005
    parser.add_argument('--l2_coef', type=float, default=0)
    parser.add_argument('--use_pos', type=bool, default=True)
    parser.add_argument('--use_normalize', type=bool, default=True)

    # model-specific parameters
    parser.add_argument('--tau', type=float, default=0.8) # 0.5
    parser.add_argument('--feat_drop', type=float, default=0.0)#0.5
    parser.add_argument('--attn_drop', type=float, default=0.0)# 0.4
    parser.add_argument('--lam', type=float, default=0.5)
    parser.add_argument('--activate', type=str, default='leaky_relu')
    parser.add_argument('--MLP_layers', type=int, default=2)
    # 数据增广
    parser.add_argument('--feat_mask_ratio1', type=float, default=0.0)
    parser.add_argument('--edge_mask_ratio1', type=float, default=0.4)#0.4  聚类0.1， 分类0.4
    parser.add_argument('--feat_mask_ratio2', type=float, default=0.0)
    parser.add_argument('--edge_mask_ratio2', type=float, default=0.4)

    # muti-view self-express network
    parser.add_argument('--lmbd', type=float, default=0.9)#0.9
    parser.add_argument('--express_layer', type=int, default=2)#2
    parser.add_argument('--express_l2', type=int, default=1e-2)# 1e-2
    parser.add_argument('--beta1', type=int, default=0.1)# 1e-2
    parser.add_argument('--beta2', type=int, default=0.1)# 1e-2

    # closed muti-view self-express
    parser.add_argument('--topk', type=int, default=8)  # 0.1 73.12 78.78
    parser.add_argument('--alpha1', type=float, default=0.1) # 0.1 73.12 78.78
    parser.add_argument('--alpha2', type=float, default=0.01) #
    parser.add_argument('--beta', type=float, default=20) # 20 71.73 77


    # filter S
    parser.add_argument('--quantile1', type=float, default=0.95)
    parser.add_argument('--quantile2', type=float, default=0.90)#0.98 0.99
    parser.add_argument('--interval', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.1)#0.2
    parser.add_argument('--sigma', type=float, default=1.0)# 聚类0.5

    args, _ = parser.parse_known_args()
    args.type_num = [6564, 13329, 35890]  # the number of every node type
    args.nei_num = 2  # the number of neighbors' types
    return args

def aminer_colsed_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--self_expressive', type=str, default='closed')  # network, closed

    parser.add_argument('--save_emb', action="store_true")
    parser.add_argument('--turn', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="aminer")
    parser.add_argument('--ratio', type=int, default=[20, 40, 60])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)#4, 0,
    parser.add_argument('--hidden_dim', type=int, default=256)#256
    parser.add_argument('--nb_epochs', type=int, default=1000)# 10000
    # The parameters of evaluation
    parser.add_argument('--eva_lr', type=float, default=0.01)
    parser.add_argument('--eva_wd', type=float, default=0)

    # The parameters of learning process
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001) # 0.005
    parser.add_argument('--l2_coef', type=float, default=0)
    parser.add_argument('--use_pos', type=bool, default=True)  # mp 增广用 True，hg 增广用 False
    parser.add_argument('--use_normalize', type=bool, default=True)  # mp 增广用 True，hg 增广用 False

    # model-specific parameters
    parser.add_argument('--tau', type=float, default=0.8) # 0.5
    parser.add_argument('--feat_drop', type=float, default=0.0)#0.5
    parser.add_argument('--attn_drop', type=float, default=0.0)# 0.4
    parser.add_argument('--lam', type=float, default=0.5)
    parser.add_argument('--activate', type=str, default='leaky_relu')
    parser.add_argument('--MLP_layers', type=int, default=2)
    # 数据增广
    parser.add_argument('--feat_mask_ratio1', type=float, default=0.0)
    parser.add_argument('--edge_mask_ratio1', type=float, default=0.4)#0.4
    parser.add_argument('--feat_mask_ratio2', type=float, default=0.0)
    parser.add_argument('--edge_mask_ratio2', type=float, default=0.4)

    # muti-view self-express network
    parser.add_argument('--lmbd', type=float, default=0.9)#0.9
    parser.add_argument('--express_layer', type=int, default=2)#2
    parser.add_argument('--express_l2', type=int, default=1e-2)# 1e-2
    parser.add_argument('--beta1', type=int, default=0.1)# 1e-2
    parser.add_argument('--beta2', type=int, default=0.1)# 1e-2

    # closed muti-view self-express
    parser.add_argument('--outlier', type=float, default=0.999)
    parser.add_argument('--topk', type=int, default=6)  # 0.1 73.12 78.78
    parser.add_argument('--alpha1', type=float, default=0.1) # 0.1 73.12 78.78
    parser.add_argument('--alpha2', type=float, default=0.1) #
    parser.add_argument('--beta', type=float, default=20) # 20 71.73 77

    # filter S
    parser.add_argument('--quantile1', type=float, default=0.95)# 0.95
    parser.add_argument('--quantile2', type=float, default=0.9)#0.98 0.99
    parser.add_argument('--interval', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.3)
    parser.add_argument('--sigma', type=float, default=0.8)#0.8

    args, _ = parser.parse_known_args()
    args.type_num = [6564, 13329, 35890]  # the number of every node type
    args.nei_num = 2  # the number of neighbors' types
    return args

def freebase_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--self_expressive', type=str, default='network')  # network, closed

    parser.add_argument('--save_emb', action="store_true")
    parser.add_argument('--turn', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="freebase")
    parser.add_argument('--ratio', type=int, default=[20, 40, 60])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=32)#32
    parser.add_argument('--hidden_dim', type=int, default=512)  # 256 小的节点分类 大的聚类
    parser.add_argument('--nb_epochs', type=int, default=10000)
    parser.add_argument('--warmup', type=int, default=0)
    parser.add_argument('--use_pos', type=bool, default=True)
    parser.add_argument('--use_normalize', type=bool, default=True)#


    # The parameters of evaluation
    parser.add_argument('--eva_lr', type=float, default=0.01)
    parser.add_argument('--eva_wd', type=float, default=0)

    # The parameters of learning process
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.0001)#0.001
    parser.add_argument('--l2_coef', type=float, default=0)


    # model-specific parameters
    parser.add_argument('--tau', type=float, default=0.5)#0.5
    parser.add_argument('--feat_drop', type=float, default=0.5)#0.1
    parser.add_argument('--attn_drop', type=float, default=0.0)
    parser.add_argument('--lam', type=float, default=0.5)
    parser.add_argument('--activate', type=str, default='leaky_relu')
    parser.add_argument('--MLP_layers', type=int, default=2)#
    # 数据增广
    parser.add_argument('--feat_mask_ratio1', type=float, default=0.0)
    parser.add_argument('--edge_mask_ratio1', type=float, default=0.4)#0.4
    parser.add_argument('--feat_mask_ratio2', type=float, default=0.0)
    parser.add_argument('--edge_mask_ratio2', type=float, default=0.4)

    # muti-view self-express network
    parser.add_argument('--lmbd', type=float, default=0.9)
    parser.add_argument('--express_layer', type=int, default=2)
    parser.add_argument('--express_l2', type=int, default=1e-2)
    parser.add_argument('--beta1', type=int, default=0.1)# 1e-2
    parser.add_argument('--beta2', type=int, default=0.1)# 1e-2

    # closed muti-view self-express
    parser.add_argument('--topk', type=int, default=5)  # 0.1 73.12 78.78
    parser.add_argument('--alpha1', type=float, default=0.1)  # 0.1 73.12 78.78
    parser.add_argument('--alpha2', type=float, default=0.01)  #
    parser.add_argument('--beta', type=float, default=20)  # 20 71.73 77

    # filter S
    parser.add_argument('--quantile1', type=float, default=0.98)
    parser.add_argument('--quantile2', type=float, default=1.0)  # 0.98 0.99
    parser.add_argument('--interval', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.1)#0.1
    parser.add_argument('--sigma', type=float, default=0.2)
    args, _ = parser.parse_known_args()
    args.type_num = [3492, 2502, 33401, 4459]  # the number of every node type
    args.nei_num = 3  # the number of neighbors' types
    return args


def freebase_closed_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--self_expressive', type=str, default='closed')  # network, closed

    parser.add_argument('--save_emb', action="store_true")
    parser.add_argument('--turn', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="freebase")
    parser.add_argument('--ratio', type=int, default=[20, 40, 60])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=32)#32
    parser.add_argument('--hidden_dim', type=int, default=1024)  # 256 大的节点分类 小的聚类
    parser.add_argument('--nb_epochs', type=int, default=10000)
    parser.add_argument('--warmup', type=int, default=0)
    parser.add_argument('--use_pos', type=bool, default=True)
    parser.add_argument('--use_normalize', type=bool, default=True)#


    # The parameters of evaluation
    parser.add_argument('--eva_lr', type=float, default=0.01)
    parser.add_argument('--eva_wd', type=float, default=0)

    # The parameters of learning process
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.0001)#0.001
    parser.add_argument('--l2_coef', type=float, default=0)


    # model-specific parameters
    parser.add_argument('--tau', type=float, default=0.5)#0.5
    parser.add_argument('--feat_drop', type=float, default=0.5)#0.1
    parser.add_argument('--attn_drop', type=float, default=0.0)
    parser.add_argument('--lam', type=float, default=0.5)
    parser.add_argument('--activate', type=str, default='leaky_relu')
    parser.add_argument('--MLP_layers', type=int, default=2)#
    # 数据增广
    parser.add_argument('--feat_mask_ratio1', type=float, default=0.0)
    parser.add_argument('--edge_mask_ratio1', type=float, default=0.5)#0.4
    parser.add_argument('--feat_mask_ratio2', type=float, default=0.0)
    parser.add_argument('--edge_mask_ratio2', type=float, default=0.5)

    # muti-view self-express network
    parser.add_argument('--lmbd', type=float, default=0.9)
    parser.add_argument('--express_layer', type=int, default=2)
    parser.add_argument('--express_l2', type=int, default=1e-2)
    parser.add_argument('--beta1', type=int, default=0.1)# 1e-2
    parser.add_argument('--beta2', type=int, default=0.1)# 1e-2

    # closed muti-view self-express
    parser.add_argument('--outlier', type=float, default=0.999)
    parser.add_argument('--topk', type=int, default=5)  # 0.1 73.12 78.78
    parser.add_argument('--alpha1', type=float, default=0.1)  # 0.1 73.12 78.78
    parser.add_argument('--alpha2', type=float, default=0.01)  #
    parser.add_argument('--beta', type=float, default=20)  # 20 71.73 77

    # filter S
    parser.add_argument('--quantile1', type=float, default=0.96)
    parser.add_argument('--quantile2', type=float, default=0.98)  # 0.98 0.99
    parser.add_argument('--interval', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.2)
    parser.add_argument('--sigma', type=float, default=0.0)
    args, _ = parser.parse_known_args()
    args.type_num = [3492, 2502, 33401, 4459]  # the number of every node type
    args.nei_num = 3  # the number of neighbors' types
    return args

def imdb_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--self_expressive', type=str, default='network')  # network, closed

    parser.add_argument('--save_emb', action="store_true")
    parser.add_argument('--turn', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="imdb")
    parser.add_argument('--ratio', type=int, default=[20, 40, 60])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=32)#32
    parser.add_argument('--hidden_dim', type=int, default=512)  # 512
    parser.add_argument('--nb_epochs', type=int, default=10000)
    parser.add_argument('--use_pos', type=bool, default=False)# False
    parser.add_argument('--use_normalize', type=bool, default=True)


    # The parameters of evaluation
    parser.add_argument('--eva_lr', type=float, default=0.01)
    parser.add_argument('--eva_wd', type=float, default=0)

    # The parameters of learning process
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001)#0.001
    parser.add_argument('--l2_coef', type=float, default=0)


    # model-specific parameters
    parser.add_argument('--tau', type=float, default=1.0)#0.8, 1.0
    parser.add_argument('--feat_drop', type=float, default=0.0) # 0.0
    parser.add_argument('--attn_drop', type=float, default=0.3) # 0.3
    parser.add_argument('--lam', type=float, default=0.5)
    parser.add_argument('--activate', type=str, default='leaky_relu')
    parser.add_argument('--MLP_layers', type=int, default=1)

    # 数据增广
    parser.add_argument('--feat_mask_ratio1', type=float, default=0.3)
    parser.add_argument('--edge_mask_ratio1', type=float, default=0.5)#0.5
    parser.add_argument('--feat_mask_ratio2', type=float, default=0.3)
    parser.add_argument('--edge_mask_ratio2', type=float, default=0.5)

    # muti-view self-express network
    parser.add_argument('--lmbd', type=float, default=0.9)
    parser.add_argument('--express_layer', type=int, default=2)# 2,3
    parser.add_argument('--express_l2', type=int, default=1e-2)
    parser.add_argument('--beta1', type=int, default=0.1)# 1e-2
    parser.add_argument('--beta2', type=int, default=0.1)# 1e-2

    # closed muti-view self-express
    parser.add_argument('--topk', type=int, default=5)  # 0.1 73.12 78.78
    parser.add_argument('--alpha1', type=float, default=0.1)  # 0.1 73.12 78.78
    parser.add_argument('--alpha2', type=float, default=0.01)  #
    parser.add_argument('--beta', type=float, default=20)  # 20 71.73 77

    # filter S
    parser.add_argument('--quantile1', type=float, default=0.8)  # 0.98
    parser.add_argument('--quantile2', type=float, default=0.85)  # 1.0
    parser.add_argument('--interval', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.1)#0.1
    parser.add_argument('--sigma', type=float, default=0.3)#0.2
    args, _ = parser.parse_known_args()
    args.type_num = [4661, 5841, 2270]  # the number of every node type
    args.nei_num = 2  # the number of neighbors' types
    return args

def imdb_closed_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--self_expressive', type=str, default='network')  # network, closed

    parser.add_argument('--save_emb', action="store_true")
    parser.add_argument('--turn', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="imdb")
    parser.add_argument('--ratio', type=int, default=[20, 40, 60])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=32)#32
    parser.add_argument('--hidden_dim', type=int, default=512)  # 512
    parser.add_argument('--nb_epochs', type=int, default=10000)
    parser.add_argument('--use_pos', type=bool, default=False)# False
    parser.add_argument('--use_normalize', type=bool, default=True)


    # The parameters of evaluation
    parser.add_argument('--eva_lr', type=float, default=0.01)
    parser.add_argument('--eva_wd', type=float, default=0)

    # The parameters of learning process
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001)#0.001
    parser.add_argument('--l2_coef', type=float, default=0)


    # model-specific parameters
    parser.add_argument('--tau', type=float, default=1.0)#0.8, 1.0
    parser.add_argument('--feat_drop', type=float, default=0.0) # 0.0
    parser.add_argument('--attn_drop', type=float, default=0.3) # 0.3
    parser.add_argument('--lam', type=float, default=0.5)
    parser.add_argument('--activate', type=str, default='leaky_relu')
    parser.add_argument('--MLP_layers', type=int, default=1)

    # 数据增广
    parser.add_argument('--feat_mask_ratio1', type=float, default=0.3)
    parser.add_argument('--edge_mask_ratio1', type=float, default=0.5)#0.5
    parser.add_argument('--feat_mask_ratio2', type=float, default=0.3)
    parser.add_argument('--edge_mask_ratio2', type=float, default=0.5)

    # muti-view self-express network
    parser.add_argument('--lmbd', type=float, default=0.9)
    parser.add_argument('--express_layer', type=int, default=2)# 2,3
    parser.add_argument('--express_l2', type=int, default=1e-2)
    parser.add_argument('--beta1', type=int, default=0.1)# 1e-2
    parser.add_argument('--beta2', type=int, default=0.1)# 1e-2

    # closed muti-view self-express
    parser.add_argument('--topk', type=int, default=5)  # 0.1 73.12 78.78
    parser.add_argument('--alpha1', type=float, default=0.1)  # 0.1 73.12 78.78
    parser.add_argument('--alpha2', type=float, default=0.01)  #
    parser.add_argument('--beta', type=float, default=20)  # 20 71.73 77

    # filter S
    parser.add_argument('--quantile1', type=float, default=0.8)  # 0.98
    parser.add_argument('--quantile2', type=float, default=0.95)  # 1.0
    parser.add_argument('--interval', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--sigma', type=float, default=0.0)#0.2
    args, _ = parser.parse_known_args()
    args.type_num = [4661, 5841, 2270]  # the number of every node type
    args.nei_num = 2  # the number of neighbors' types
    return args

def academic_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--self_expressive', type=str, default='network')  # network, closed

    parser.add_argument('--save_emb', action="store_true")
    parser.add_argument('--turn', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="academic")
    parser.add_argument('--ratio', type=int, default=[20, 40, 60])
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--seed', type=int, default=32)#32
    parser.add_argument('--hidden_dim', type=int, default=512)  # 256 小的节点分类 大的聚类
    parser.add_argument('--nb_epochs', type=int, default=10000)
    parser.add_argument('--use_pos', type=bool, default=True)
    parser.add_argument('--use_normalize', type=bool, default=True)


    # The parameters of evaluation
    parser.add_argument('--eva_lr', type=float, default=0.01)
    parser.add_argument('--eva_wd', type=float, default=0)

    # The parameters of learning process
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001)#0.001
    parser.add_argument('--l2_coef', type=float, default=0)


    # model-specific parameters
    parser.add_argument('--tau', type=float, default=0.6)#0.5
    parser.add_argument('--feat_drop', type=float, default=0.0)#0.0
    parser.add_argument('--attn_drop', type=float, default=0.3) # 0.3
    parser.add_argument('--lam', type=float, default=0.5)
    parser.add_argument('--activate', type=str, default='leaky_relu')
    parser.add_argument('--MLP_layers', type=int, default=2)

    # 数据增广
    parser.add_argument('--feat_mask_ratio1', type=float, default=0.2)
    parser.add_argument('--edge_mask_ratio1', type=float, default=0.5)#0.5
    parser.add_argument('--feat_mask_ratio2', type=float, default=0.2)
    parser.add_argument('--edge_mask_ratio2', type=float, default=0.5)

    # muti-view self-express network
    parser.add_argument('--lmbd', type=float, default=0.9)
    parser.add_argument('--express_layer', type=int, default=2)
    parser.add_argument('--express_l2', type=int, default=1e-2)
    parser.add_argument('--beta1', type=int, default=0.1)# 1e-2
    parser.add_argument('--beta2', type=int, default=0.1)# 1e-2

    # closed muti-view self-express
    parser.add_argument('--topk', type=int, default=5)  # 0.1 73.12 78.78
    parser.add_argument('--alpha1', type=float, default=0.1)  # 0.1 73.12 78.78
    parser.add_argument('--alpha2', type=float, default=0.01)  #
    parser.add_argument('--beta', type=float, default=20)  # 20 71.73 77

    # filter S
    parser.add_argument('--quantile1', type=float, default=0.98)
    parser.add_argument('--quantile2', type=float, default=1.0)  # 0.98 0.99
    parser.add_argument('--interval', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--sigma', type=float, default=0.2)
    args, _ = parser.parse_known_args()
    args.type_num = [4661, 5841, 2270]  # the number of every node type
    args.nei_num = 2  # the number of neighbors' types
    return args

def academic_closed_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--self_expressive', type=str, default='closed')  # network, closed

    parser.add_argument('--save_emb', action="store_true")
    parser.add_argument('--turn', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="academic")
    parser.add_argument('--ratio', type=int, default=[20, 40, 60])
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--seed', type=int, default=32)#32
    parser.add_argument('--hidden_dim', type=int, default=512)  # 256 小的节点分类 大的聚类
    parser.add_argument('--nb_epochs', type=int, default=10000)
    parser.add_argument('--use_pos', type=bool, default=True)
    parser.add_argument('--use_normalize', type=bool, default=True)


    # The parameters of evaluation
    parser.add_argument('--eva_lr', type=float, default=0.01)
    parser.add_argument('--eva_wd', type=float, default=0)

    # The parameters of learning process
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001)#0.001
    parser.add_argument('--l2_coef', type=float, default=0)


    # model-specific parameters
    parser.add_argument('--tau', type=float, default=0.6)#0.5
    parser.add_argument('--feat_drop', type=float, default=0.0)#0.0
    parser.add_argument('--attn_drop', type=float, default=0.3) # 0.3
    parser.add_argument('--lam', type=float, default=0.5)
    parser.add_argument('--activate', type=str, default='leaky_relu')
    parser.add_argument('--MLP_layers', type=int, default=2)

    # 数据增广
    parser.add_argument('--feat_mask_ratio1', type=float, default=0.2)
    parser.add_argument('--edge_mask_ratio1', type=float, default=0.5)#0.5
    parser.add_argument('--feat_mask_ratio2', type=float, default=0.2)
    parser.add_argument('--edge_mask_ratio2', type=float, default=0.5)

    # muti-view self-express network
    parser.add_argument('--lmbd', type=float, default=0.9)
    parser.add_argument('--express_layer', type=int, default=2)
    parser.add_argument('--express_l2', type=int, default=1e-2)
    parser.add_argument('--beta1', type=int, default=0.1)# 1e-2
    parser.add_argument('--beta2', type=int, default=0.1)# 1e-2

    # closed muti-view self-express
    parser.add_argument('--topk', type=int, default=5)  # 0.1 73.12 78.78
    parser.add_argument('--alpha1', type=float, default=0.1)  # 0.1 73.12 78.78
    parser.add_argument('--alpha2', type=float, default=0.01)  #0.01
    parser.add_argument('--beta', type=float, default=20)  # 20 71.73 77

    # filter S
    parser.add_argument('--outlier', type=float, default=0.999)
    parser.add_argument('--quantile1', type=float, default=0.95)
    parser.add_argument('--quantile2', type=float, default=0.85)  # 0.98 0.99
    parser.add_argument('--interval', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.05)
    parser.add_argument('--sigma', type=float, default=0.2)
    args, _ = parser.parse_known_args()
    args.type_num = [4661, 5841, 2270]  # the number of every node type
    args.nei_num = 2  # the number of neighbors' types
    return args


def yelp_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--self_expressive', type=str, default='network')  # network, closed

    parser.add_argument('--save_emb', action="store_true")
    parser.add_argument('--turn', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="yelp")
    parser.add_argument('--ratio', type=int, default=[20, 40, 60])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=32)#32
    parser.add_argument('--hidden_dim', type=int, default=512)  # 256 小的节点分类 大的聚类
    parser.add_argument('--nb_epochs', type=int, default=10000)
    parser.add_argument('--use_pos', type=bool, default=False) # True聚类更好
    parser.add_argument('--use_normalize', type=bool, default=True)

    # The parameters of evaluation
    parser.add_argument('--eva_lr', type=float, default=0.01)
    parser.add_argument('--eva_wd', type=float, default=0)

    # The parameters of learning process
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.0001)#0.001
    parser.add_argument('--l2_coef', type=float, default=1e-5)


    # model-specific parameters
    parser.add_argument('--tau', type=float, default=0.5)#0.5
    parser.add_argument('--feat_drop', type=float, default=0.0)
    parser.add_argument('--attn_drop', type=float, default=0.3) # 0.3
    parser.add_argument('--lam', type=float, default=0.5)
    parser.add_argument('--activate', type=str, default='leaky_relu')
    parser.add_argument('--MLP_layers', type=int, default=1)

    # 数据增广
    parser.add_argument('--feat_mask_ratio1', type=float, default=0.4)
    parser.add_argument('--edge_mask_ratio1', type=float, default=0.5)#0.5
    parser.add_argument('--feat_mask_ratio2', type=float, default=0.4)
    parser.add_argument('--edge_mask_ratio2', type=float, default=0.5)

    # muti-view self-express network
    parser.add_argument('--lmbd', type=float, default=0.9)
    parser.add_argument('--express_layer', type=int, default=2)
    parser.add_argument('--express_l2', type=int, default=1e-2)
    parser.add_argument('--beta1', type=int, default=0.1)# 1e-2
    parser.add_argument('--beta2', type=int, default=0.1)# 1e-2

    # closed muti-view self-express
    parser.add_argument('--topk', type=int, default=5)  # 0.1 73.12 78.78
    parser.add_argument('--alpha1', type=float, default=0.1)  # 0.1 73.12 78.78
    parser.add_argument('--alpha2', type=float, default=0.01)  #
    parser.add_argument('--beta', type=float, default=20)  # 20 71.73 77

    # filter S
    parser.add_argument('--quantile1', type=float, default=0.98)
    parser.add_argument('--quantile2', type=float, default=1.0)  # 0.98 0.99
    parser.add_argument('--interval', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--sigma', type=float, default=0.2)
    args, _ = parser.parse_known_args()
    args.type_num = [2614, 1286, 4, 9]  # the number of every node type
    args.nei_num = 3 # the number of neighbors' types
    return args

def set_params():
    if dataset == "acm":
        if self_express == 'network': args = acm_params()
        else: args = acm_closed_params()
    elif dataset == "dblp":
        if self_express == 'network': args = dblp_params()
        else: args = dblp_closed_params()
    elif dataset == "aminer":
        if self_express == 'network': args = aminer_params()
        else: args = aminer_colsed_params()
    elif dataset == "freebase":
        if self_express == 'network': args = freebase_params()
        else: args = freebase_closed_params()
    elif dataset == "imdb":
        if self_express == 'network': args = imdb_params()
        else: args = imdb_closed_params()
    elif dataset == "academic":
        if self_express == 'network': args = academic_params()
        else: args = academic_closed_params()
    return args
