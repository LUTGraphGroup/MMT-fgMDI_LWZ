import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from scipy import interp
from sklearn.decomposition import PCA
import torch
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')


def load_data(seed, n_components, pe_dim, hops):
    # 读取不同元路径下的语义网络
    Dr_M = pd.read_csv('../data/Dr-M.csv', header=0).values
    Dr_Down_M = pd.read_csv('../data/Dr-Down-M.csv', header=0).values
    Dr_Up_M = pd.read_csv('../data/Dr-Up-M.csv', header=0).values
    Dr_Di = pd.read_csv('../data/Dr-Di.csv', header=0).values
    Dr_G = pd.read_csv('../data/Dr-G.csv', header=0).values
    M_Di = pd.read_csv('../data/M-Di.csv', header=0).values
    M_G = pd.read_csv('../data/M-G.csv', header=0).values
    Drug_mol2vec = pd.read_csv('../data/drug_mol2vec.csv', header=0).values
    Meta_mol2vec = pd.read_csv('../data/metabolite_mol2vec.csv', header=0).values
    Dis_MESH2vec = pd.read_csv('../data/Mesh2vec.csv', header=0).values
    Gene_2vec = pd.read_csv('../data/DNA2vec.csv', header=0).values

    pca = PCA(n_components=n_components)
    PCA_dis_feature = pca.fit_transform(Dis_MESH2vec)
    PCA_meta_feature = pca.fit_transform(Meta_mol2vec)
    PCA_drug_feature = pca.fit_transform(Drug_mol2vec)
    PCA_gene_feature = pca.fit_transform(Gene_2vec)
    Dis_feature = torch.FloatTensor(PCA_dis_feature).to(device)
    Meta_feature = torch.FloatTensor(PCA_meta_feature).to(device)
    Drug_feature = torch.FloatTensor(PCA_drug_feature).to(device)
    Gene_feature = torch.FloatTensor(PCA_gene_feature).to(device)

    Dr_Di_feature = torch.cat((Drug_feature, Dis_feature), dim=0).to(device)
    Dr_G_feature = torch.cat((Drug_feature, Gene_feature), dim=0).to(device)
    M_Di_feature = torch.cat((Meta_feature, Dis_feature), dim=0).to(device)
    M_G_feature = torch.cat((Meta_feature, Gene_feature), dim=0).to(device)

    Dr_Di_centrality = pd.read_csv('../data/Dr-Di_centrality.csv', header=None).values
    Dr_Di_centrality = torch.tensor(Dr_Di_centrality).to(device)
    Dr_G_centrality = pd.read_csv('../data/Dr-G_centrality.csv', header=None).values
    Dr_G_centrality = torch.tensor(Dr_G_centrality).to(device)
    M_Di_centrality = pd.read_csv('../data/M-Di_centrality.csv', header=None).values
    M_Di_centrality = torch.tensor(M_Di_centrality).to(device)
    M_G_centrality = pd.read_csv('../data/M-G_centrality.csv', header=None).values
    M_G_centrality = torch.tensor(M_G_centrality).to(device)

    Dr_Down_M_adj = constructNet(torch.tensor(Dr_Down_M)).to(device)
    Dr_Up_M_adj = constructNet(torch.tensor(Dr_Up_M)).to(device)

    Dr_Di_adj = constructNet(torch.tensor(Dr_Di)).to(device)
    lpe_Dr_Di = laplacian_positional_encoding(Dr_Di_adj, pe_dim).to(device)

    Dr_G_adj = constructNet(torch.tensor(Dr_G)).to(device)
    lpe_Dr_G = laplacian_positional_encoding(Dr_G_adj, pe_dim).to(device)

    M_Di_adj = constructNet(torch.tensor(M_Di)).to(device)
    lpe_M_Di = laplacian_positional_encoding(M_Di_adj, pe_dim).to(device)

    M_G_adj = constructNet(torch.tensor(M_G)).to(device)
    lpe_M_G = laplacian_positional_encoding(M_G_adj, pe_dim).to(device)

    Dr_Di_input = torch.cat((Dr_Di_feature, Dr_Di_centrality, lpe_Dr_Di), dim=1).to(device)
    Dr_G_input = torch.cat((Dr_G_feature, Dr_G_centrality, lpe_Dr_G), dim=1).to(device)
    M_Di_input = torch.cat((M_Di_feature, M_Di_centrality, lpe_M_Di), dim=1).to(device)
    M_G_input = torch.cat((M_G_feature, M_G_centrality, lpe_M_G), dim=1).to(device)

    Dr_Di_hops = re_features(Dr_Di_adj, Dr_Di_input, hops).to(device)
    Dr_G_hops = re_features(Dr_G_adj, Dr_G_input, hops).to(device)
    M_Di_hops = re_features(M_Di_adj, M_Di_input, hops).to(device)
    M_G_hops = re_features(M_G_adj, M_G_input, hops).to(device)

    label_matrix = pd.read_csv('../data/label_matrix.csv', header=0).values
    index_matrix = np.mat(np.where(label_matrix == 1))
    association_nam = index_matrix.shape[1]
    random_index = index_matrix.T.tolist()
    random.seed(seed)
    random.shuffle(random_index)
    k_folds = 5
    CV_size = int(association_nam / k_folds)
    temp = np.array(random_index[:association_nam - association_nam %
                                  k_folds]).reshape(k_folds, CV_size, -1).tolist()
    temp[k_folds - 1] = temp[k_folds - 1] + \
                        random_index[association_nam - association_nam % k_folds:]
    label1_index = temp


    index_matrix = np.mat(np.where(label_matrix == 2))
    association_nam = index_matrix.shape[1]
    random_index = index_matrix.T.tolist()
    random.seed(seed)
    random.shuffle(random_index)
    k_folds = 5
    CV_size = int(association_nam / k_folds)
    temp = np.array(random_index[:association_nam - association_nam %
                                  k_folds]).reshape(k_folds, CV_size, -1).tolist()
    temp[k_folds - 1] = temp[k_folds - 1] + \
                        random_index[association_nam - association_nam % k_folds:]
    label2_index = temp
    return Dr_M, Dr_Down_M_adj, Dr_Up_M_adj, Dr_Di_hops, Dr_G_hops, M_Di_hops, M_G_hops, label_matrix, label1_index, label2_index, k_folds


def constructNet(association_matrix):
    n, m = association_matrix.shape
    drug_matrix = torch.zeros((n, n), dtype=torch.int8)
    meta_matrix = torch.zeros((m, m), dtype=torch.int8)
    mat1 = torch.cat((drug_matrix, association_matrix), dim=1)
    mat2 = torch.cat((association_matrix.t(), meta_matrix), dim=1)
    adj_0 = torch.cat((mat1, mat2), dim=0)
    return adj_0


def laplacian_positional_encoding(adj, pe_dim):
    D = torch.diag(torch.sum(adj, dim=1))
    N = torch.diag(torch.pow(torch.sum(adj, dim=1).clamp(min=1), -0.5))
    L = torch.eye(adj.shape[0]).to(device) - N @ adj @ N
    EigVal, EigVec = torch.linalg.eig(L)
    EigVal = EigVal.real
    EigVec = EigVec.real
    sorted_indices = EigVal.argsort()
    EigVal_sorted = EigVal[sorted_indices]
    EigVec_sorted = EigVec[:, sorted_indices]
    lap_pos_enc = (EigVec_sorted[:, 1:pe_dim + 1]).float()
    return lap_pos_enc


def re_features(adj, features, K):
    nodes_features = torch.empty(features.shape[0], 1, K+1, features.shape[1])
    for i in range(features.shape[0]):
        nodes_features[i, 0, 0, :] = features[i]
    x = features + torch.zeros_like(features)
    x = x.double()
    for i in range(K):
        x = torch.matmul(adj, x)
        for index in range(features.shape[0]):
            nodes_features[index, 0, i + 1, :] = x[index]
    nodes_features = nodes_features.squeeze()
    return nodes_features


# Confusion Matrix Visualization
def plot_cm(average_cm, directory, name):
    # plt.figure(figsize=(8, 6))
    sns.heatmap(average_cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=[0, 1, 2], yticklabels=[0, 1, 2])
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14)
    plt.savefig(directory + '/%s.pdf' % name, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


'''可视化分类结果'''
def plot_T_SNE(tsne_scores_result, tsne_labels_result, directory, name):
    scaler = StandardScaler()
    tsne_scores_result = scaler.fit_transform(tsne_scores_result)
    tsne = TSNE(n_components=2, random_state=42, perplexity=50)
    X_test_tsne = tsne.fit_transform(tsne_scores_result)

    class_labels = {0: 'NR', 1: 'UR', 2: 'DR'}
    tsne_labels_result = [class_labels[y] for y in tsne_labels_result]

    # plt.figure(figsize=(10, 7))
    sns.scatterplot(x=X_test_tsne[:, 0], y=X_test_tsne[:, 1], hue=tsne_labels_result, palette='bright', s=60)
    plt.title('T_SNE Visualization of MMT-fgMDI', fontsize=14)
    # plt.xlabel('t-SNE Component 1')
    # plt.ylabel('t-SNE Component 2')
    plt.savefig(directory + '/%s.pdf' % name, dpi=300, bbox_inches='tight')
    plt.legend()
    plt.show()
