import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU()

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight, gain=1.414)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj):
        seq_fts = self.fc(seq)
        out = torch.spmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        return self.act(out)



class GConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(GConv, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = nn.PReLU(hidden_dim)
        for i in range(num_layers):
            if i == 0:
                self.layers.append(GCN(input_dim, hidden_dim))
            else:
                self.layers.append(GCN(hidden_dim, hidden_dim))

    def forward(self, x, edge_index, edge_weight=None):
        z = x
        zs = []
        for conv in self.layers:
            z = conv(z, edge_index)
            # z = self.activation(z) # 似乎不加更好
            zs.append(z)
        z = zs[-1]
        return z

class Attention(nn.Module):
    def __init__(self, hidden_dim, attn_drop):
        super(Attention, self).__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias=True)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)

        self.tanh = nn.Tanh()
        self.att = nn.Parameter(torch.empty(size=(1, hidden_dim)), requires_grad=True)
        nn.init.xavier_normal_(self.att.data, gain=1.414)

        self.softmax = nn.Softmax()
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x

    def forward(self, embeds):
        beta = []
        attn_curr = self.attn_drop(self.att)
        for embed in embeds:
            sp = self.tanh(self.fc(embed)).mean(dim=0)
            beta.append(attn_curr.matmul(sp.t()))
        beta = torch.cat(beta, dim=-1).view(-1)
        beta = self.softmax(beta)
        # print("mp ", beta.data.cpu().numpy())  # semantic attention
        z_mp = 0
        for i in range(len(embeds)):
            z_mp += embeds[i]*beta[i]
        return z_mp, beta

# class MLP(nn.Module):
#     def __init__(self, input_dim, hid_dim, output_dim):
#         super(MLP, self).__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(input_dim, hid_dim),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(hid_dim, output_dim)
#         )
#
#     def forward(self, x):
#         return self.fc(x)

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
        # x = self.bn0(x)
        for lin, bn in zip(self.lins[:-1], self.bns[:-1]):
            x = lin(x)
            x = self.activate(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            # x = bn(x)
        x = self.lins[-1](x)
        if self.last_activate:
            x = self.activate(x)
        return x

class Mp_encoder(nn.Module):
    def __init__(self, P, hidden_dim, MLP_layers, activate, feat_drop, attn_drop, input_dim=None):
        super(Mp_encoder, self).__init__()
        self.P = P

        if feat_drop:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x: x
        self.node_level = nn.ModuleList([GConv(hidden_dim, hidden_dim, 1) for _ in range(P)])
        # self.node_level = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(P)])
        # self.node_level = nn.ModuleList([GATConv(hidden_dim, hidden_dim, 2, False) for _ in range(P)])
        self.att = Attention(hidden_dim, attn_drop)

        # self.fc = nn.Linear(input_dim, hidden_dim, bias=True)
        self.fc = MLP(input_dim, hidden_dim, MLP_layers, feat_drop, activate)

    def forward(self, h, mps, fc=False, return_list=False):
        if fc == True:
            h = self.fc(h)
        embeds = []
        for i in range(self.P):
            embeds.append(self.node_level[i](h, mps[i]))
            # embeds.append(self.node_level[i](h))
        z_mp, beta = self.att(embeds)
        if return_list == True:
            return z_mp, embeds, h, beta
        else:
            return z_mp

class Mp_encoder_dgl(nn.Module):
    def __init__(self, P, hidden_dim, attn_drop):
        super(Mp_encoder_dgl, self).__init__()
        self.P = P
        # self.node_level = nn.ModuleList([GCN(hidden_dim, hidden_dim) for _ in range(P)])
        self.node_level = nn.ModuleList([GConv(hidden_dim, hidden_dim, 1) for _ in range(P)])
        self.att = Attention(hidden_dim, attn_drop)

    def forward(self, h, mps, fc=False):
        embeds = []
        for i in range(self.P):
            embeds.append(self.node_level[i](h, mps[i]))
        z_mp = self.att(embeds)
        return z_mp