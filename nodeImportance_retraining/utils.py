import os
import torch
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from gcn import GCN, Target_GCN
from gat import GAT, Target_GAT
from tagcn import TAGCN, Target_TAGCN
from graphsage import GraphSAGE, Target_GraphSAGE
from scipy.stats import entropy
from collections import Counter


def get_model_path(path_dir_compile):
    model_path_list = []
    if os.path.isdir(path_dir_compile):
        for root, dirs, files in os.walk(path_dir_compile, topdown=True):
            for file in files:
                file_absolute_path = os.path.join(root, file)
                if file_absolute_path.endswith('.pt'):
                    model_path_list.append(file_absolute_path)
    return model_path_list


def load_model(model_name, path_model, hidden_channel, num_node_features, num_classes, dic):
    if model_name == 'gcn':
        model = GCN(num_node_features, hidden_channel, num_classes, dic)
    elif model_name == 'gat':
        model = GAT(num_node_features, hidden_channel, num_classes, dic)
    elif model_name == 'graphsage':
        model = GraphSAGE(num_node_features, hidden_channel, num_classes, dic)
    elif model_name == 'tagcn':
        model = TAGCN(num_node_features, hidden_channel, num_classes, dic)
    model.load_state_dict(torch.load(path_model, map_location=torch.device('cpu')))
    model.eval()
    return model


def load_target_model(model_name, num_node_features, hidden_channel, num_classes, target_model_path):
    if model_name == 'gcn':
        model = Target_GCN(num_node_features, hidden_channel, num_classes)
    elif model_name == 'gat':
        model = Target_GAT(num_node_features, hidden_channel, num_classes)
    elif model_name == 'graphsage':
        model = Target_GraphSAGE(num_node_features, hidden_channel, num_classes)
    elif model_name == 'tagcn':
        model = Target_TAGCN(num_node_features, hidden_channel, num_classes)
    model.load_state_dict(torch.load(target_model_path, map_location=torch.device('cpu')))
    model.eval()
    return model


def select_model(model_name, hidden_channel, num_node_features, num_classes, dic):
    if model_name == 'gcn':
        model = GCN(num_node_features, hidden_channel, num_classes, dic)
    elif model_name == 'gat':
        model = GAT(num_node_features, hidden_channel, num_classes, dic)
    elif model_name == 'graphsage':
        model = GraphSAGE(num_node_features, hidden_channel, num_classes, dic)
    elif model_name == 'tagcn':
        model = TAGCN(num_node_features, hidden_channel, num_classes, dic)
    return model


def get_res_ratio_list(idx_miss_list, select_idx_list, select_ratio_list):
    res_ratio_list = []
    for i in select_ratio_list:
        n = round(len(select_idx_list) * i)
        tmp_select_idx_list = select_idx_list[: n]
        n_hit = len(np.intersect1d(idx_miss_list, tmp_select_idx_list, assume_unique=False, return_indices=False))
        ratio = round(n_hit / len(idx_miss_list), 4)
        res_ratio_list.append(ratio)
    return res_ratio_list


def load_data(path_x_np, path_edge_index, path_y):
    x = pickle.load(open(path_x_np, 'rb'))
    edge_index = pickle.load(open(path_edge_index, 'rb'))
    y = pickle.load(open(path_y, 'rb'))

    num_node_features = len(x[0])
    num_classes = len(set(y))
    idx_np = np.array(list(range(len(x))))
    train_idx, test_idx, train_y, test_y = train_test_split(idx_np, y, test_size=0.2, random_state=17)

    x = torch.from_numpy(x)
    edge_index = torch.from_numpy(edge_index)
    y = torch.from_numpy(y)

    return num_node_features, num_classes, x, edge_index, y, test_y, train_idx, test_idx


# edge_index_np.pkl   (2, 10556)
# [[   0    0    0 ... 2707 2707 2707]
#  [ 633 1862 2582 ...  598 1473 2706]]

# to

# modified_adj
# [[0., 0., 0.,  ..., 0., 0., 0.],
#         [0., 0., 0.,  ..., 0., 0., 0.],
#         [0., 0., 0.,  ..., 0., 0., 0.],
#         ...,
#         [0., 0., 0.,  ..., 0., 0., 0.],
#         [0., 0., 0.,  ..., 0., 0., 0.],
#         [0., 0., 0.,  ..., 0., 0., 0.]]


def edge_index_to_adj(edge_index_np):
    n_node = max(edge_index_np[0])+1
    m = np.full((n_node, n_node), 0)
    i_j_list = []
    for idx in range(len(edge_index_np[0])):
        i = edge_index_np[0][idx]
        j = edge_index_np[1][idx]
        if [i, j] not in i_j_list and [j, i] not in i_j_list:
            i_j_list.append([i, j])

    for v in i_j_list:
        i = v[0]
        j = v[1]
        m[i][j] = 1
        m[j][i] = 1
    return m


def adj_to_edge_index(adj):
    n_node = len(adj)
    up_list = []
    down_list = []
    for i in range(n_node):
        for j in range(n_node):
            if adj[i][j]==1:
                up_list.append(i)
                down_list.append(j)
                up_list.append(j)
                down_list.append(i)
    m = np.array([up_list, down_list])
    return m


def get_shannon(value_list):
    """return Shannon Index"""
    sample_counts = Counter(value_list)
    sample_probs = np.array(list(sample_counts.values())) / len(value_list)
    shannon_index = entropy(sample_probs, base=np.e)
    return shannon_index


def get_hill(value_list):
    """return Hill Numbers"""
    sample_counts = Counter(value_list)
    sample_probs = np.array(list(sample_counts.values())) / len(value_list)
    hill_number = (np.sum(sample_probs**2))**(1/(1-2))
    return hill_number

