import numpy as np
import torch
from .logreg import LogReg
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.nn.functional import softmax
from sklearn.metrics import roc_auc_score
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
import matplotlib
# matplotlib.use('TkAgg')
# matplotlib.use('Agg')  # 使用Agg后端，这个后端适用于生成图像文件但不显示它们
import matplotlib.pyplot as plt
import numpy as np

def evaluate(embeds, ratio, idx_train, idx_val, idx_test, label, nb_classes, device, dataset, lr, wd
             , isTest=True):
    hid_units = embeds.shape[1]
    xent = nn.CrossEntropyLoss()

    train_embs = embeds[idx_train]
    val_embs = embeds[idx_val]
    test_embs = embeds[idx_test]

    train_lbls = torch.argmax(label[idx_train], dim=-1)
    val_lbls = torch.argmax(label[idx_val], dim=-1)
    test_lbls = torch.argmax(label[idx_test], dim=-1)
    accs = []
    micro_f1s = []
    macro_f1s = []
    macro_f1s_val = []
    auc_score_list = []
    # 50
    for _ in range(5):
        log = LogReg(hid_units, nb_classes)
        opt = torch.optim.Adam(log.parameters(), lr=lr, weight_decay=wd)
        log.to(device)

        val_accs = []
        test_accs = []
        val_micro_f1s = []
        test_micro_f1s = []
        val_macro_f1s = []
        test_macro_f1s = []

        logits_list = []
        for iter_ in range(250):
            # train
            log.train()
            opt.zero_grad()

            logits = log(train_embs)
            loss = xent(logits, train_lbls)

            loss.backward(retain_graph=True)
            opt.step()

            # val
            logits = log(val_embs)
            preds = torch.argmax(logits, dim=1)

            val_acc = torch.sum(preds == val_lbls).float() / val_lbls.shape[0]
            val_f1_macro = f1_score(val_lbls.cpu(), preds.cpu(), average='macro')
            val_f1_micro = f1_score(val_lbls.cpu(), preds.cpu(), average='micro')

            val_accs.append(val_acc.item())
            val_macro_f1s.append(val_f1_macro)
            val_micro_f1s.append(val_f1_micro)

            # test
            logits = log(test_embs)
            preds = torch.argmax(logits, dim=1)

            test_acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
            test_f1_macro = f1_score(test_lbls.cpu(), preds.cpu(), average='macro')
            test_f1_micro = f1_score(test_lbls.cpu(), preds.cpu(), average='micro')

            test_accs.append(test_acc.item())
            test_macro_f1s.append(test_f1_macro)
            test_micro_f1s.append(test_f1_micro)
            logits_list.append(logits)

        max_iter = val_accs.index(max(val_accs))
        accs.append(test_accs[max_iter])
        max_iter = val_macro_f1s.index(max(val_macro_f1s))
        macro_f1s.append(test_macro_f1s[max_iter])
        macro_f1s_val.append(val_macro_f1s[max_iter])

        max_iter = val_micro_f1s.index(max(val_micro_f1s))
        micro_f1s.append(test_micro_f1s[max_iter])

        # auc
        best_logits = logits_list[max_iter]
        best_proba = softmax(best_logits, dim=1)
        auc_score_list.append(roc_auc_score(y_true=test_lbls.detach().cpu().numpy(),
                                            y_score=best_proba.detach().cpu().numpy(),
                                            multi_class='ovr'
                                            ))

    if isTest:
        print("\t[Class {}] Macro-F1_mean: {:.4f} var: {:.4f}  Micro-F1_mean: {:.4f} var: {:.4f} auc {:.4f}"
              .format(ratio,
                      np.mean(macro_f1s),
                      np.std(macro_f1s),
                      np.mean(micro_f1s),
                      np.std(micro_f1s),
                      np.mean(auc_score_list),
                      np.std(auc_score_list)
                      )
              )
    else:
        return np.mean(macro_f1s_val), np.mean(macro_f1s)

    # f = open("result_"+dataset+str(ratio)+".txt", "a")
    # f.write(str(np.mean(macro_f1s))+"\t"+str(np.mean(micro_f1s))+"\t"+str(np.mean(auc_score_list))+"\n")
    # f.close()

class MLP(nn.Module):
    def __init__(self, input_dim, hid_dim, output_dim):
        super(MLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hid_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hid_dim, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

def evaluate_mp_lp(h_src, h_dst, sub_g, hid_dim, device, lr, wd, model_name):
    # 50
    for _ in range(5):
        pos_edges = sub_g.adj().coalesce().indices().to(device)
        neg_src, neg_dst = sub_g.global_uniform_negative_sampling(pos_edges.size(1))
        neg_edges = torch.stack([neg_src, neg_dst], dim=0)

        all_edges = torch.cat([pos_edges, neg_edges], dim=1)
        pos_lbl, neg_lbl = torch.ones(pos_edges.size(1), dtype=int), torch.zeros(neg_edges.size(1), dtype=int)
        lbl = torch.cat([pos_lbl, neg_lbl], dim=0).cuda()
        edge_embed = torch.cat([h_src[all_edges[0]], h_dst[all_edges[1]]], dim=1)
        embeds, label = edge_embed, lbl
        edge_num = all_edges.shape[1]
        indice = np.arange(edge_num)
        idx_train, idx_test, _, _ = train_test_split(indice, indice, test_size=0.8, random_state=2024, shuffle=True)
        idx_train, idx_val = idx_train[:int(len(idx_train) * 0.5)], idx_train[int(len(idx_train) * 0.5):]
        hid_units = embeds.shape[1]
        xent = nn.CrossEntropyLoss()

        train_embs = embeds[idx_train]
        val_embs = embeds[idx_val]
        test_embs = embeds[idx_test]

        train_lbls = label[idx_train]
        val_lbls = label[idx_val]
        test_lbls = label[idx_test]
        accs = []
        micro_f1s = []
        macro_f1s = []
        macro_f1s_val = []
        auc_score_list = []

        # acm: 512,
        mlp = MLP(hid_units, hid_dim, 2)
        opt = torch.optim.Adam(mlp.parameters(), lr=lr, weight_decay=wd)
        mlp.to(device)

        val_accs = []
        test_accs = []
        val_micro_f1s = []
        test_micro_f1s = []
        val_macro_f1s = []
        test_macro_f1s = []

        logits_list = []
        best = 0.0
        cnt_wait = 0
        for epoch in range(1000):
            # train
            mlp.train()
            opt.zero_grad()

            logits = mlp(train_embs)
            loss = xent(logits, train_lbls)

            loss.backward(retain_graph=True)
            opt.step()
            torch.cuda.empty_cache()
            # val
            mlp.eval()
            logits = mlp(val_embs)
            preds = torch.argmax(logits, dim=1)

            val_acc = torch.sum(preds == val_lbls).float() / val_lbls.shape[0]
            # val_f1_macro = f1_score(val_lbls.cpu(), preds.cpu(), average='macro')
            # val_f1_micro = f1_score(val_lbls.cpu(), preds.cpu(), average='micro')
            if val_acc > best:
                best = val_acc
                cnt_wait = 0
                torch.save(mlp.state_dict(), model_name + '_MLP_LP.pkl')
            else:
                cnt_wait += 1

            if cnt_wait == 50:
                # torch.save(best_model, 'HDMI_' + own_str + '.pkl')
                # print('Early stopping!')
                break
        mlp.load_state_dict(torch.load(model_name + '_MLP_LP.pkl'))
        mlp.eval()
        # test
        logits = mlp(test_embs)
        preds = torch.argmax(logits, dim=1)
        test_acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
        test_f1_macro = f1_score(test_lbls.cpu(), preds.cpu(), average='macro')
        test_f1_micro = f1_score(test_lbls.cpu(), preds.cpu(), average='micro')

        test_accs.append(test_acc.item())
        macro_f1s.append(test_f1_macro)
        micro_f1s.append(test_f1_micro)
        logits_list.append(logits)
        # auc
        best_logits = logits#logits_list[max_iter]
        best_proba = softmax(best_logits, dim=1)
        auc_score_list.append(roc_auc_score(y_true=test_lbls.detach().cpu().numpy(),
                                            y_score=best_proba.argmax(1).detach().cpu().numpy(),
                                            multi_class='ovr'
                                            ))


    print("\t[He_link prediction] Macro-F1_mean: {:.4f} var: {:.4f}  Micro-F1_mean: {:.4f} var: {:.4f} auc {:.4f}"
          .format(np.mean(macro_f1s),
                  np.std(macro_f1s),
                  np.mean(micro_f1s),
                  np.std(micro_f1s),
                  np.mean(auc_score_list),
                  np.std(auc_score_list)
                  )
          )

def show_cluster_vis(embeds, label, own_str, nb_classes):
    # pca = PCA(n_components=2)
    # reduced_features = pca.fit_transform(embeds)
    tsne = TSNE(n_components=2, init='pca')
    reduced_features = tsne.fit_transform(embeds)

    Y_pred = KMeans(nb_classes, random_state=42).fit(embeds)#.predict(embeds)
    Y_pred = Y_pred.labels_
    # 可视化
    label_to_color = {0: '#FF9671', 1: '#008E9B', 2: '#B39CD0', 3: 'red', 4:'blue'}  # 标签到颜色的映射关系
    colors = [label_to_color[l] for l in label]  # 根据标签获取颜色
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=colors, alpha=0.7, s=10)

    score = silhouette_score(embeds, Y_pred)
    print(f"Silhouette score: {score:.4f}")
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('./figures' + '_' + own_str + 'clustering.png', dpi=600)
    plt.show()


def evaluate_cluster(embeds, y, n_labels, kmeans_random_state):
    Y_pred = KMeans(n_labels, random_state=kmeans_random_state).fit(embeds)#.predict(embeds)
    Y_pred = Y_pred.labels_

    # Y_pred = SpectralClustering(n_clusters=n_labels, affinity='nearest_neighbors').fit(embeds)
    # Y_pred = Y_pred.labels_

    nmi = normalized_mutual_info_score(y, Y_pred)
    ari = adjusted_rand_score(y, Y_pred)
    return nmi, ari

