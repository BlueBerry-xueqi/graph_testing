import torch
import pickle
import numpy as np
import json
from sklearn.model_selection import train_test_split
from edge_models import GraphSAGEdge, TAGCNEdge
from sklearn.metrics import accuracy_score
from get_rank_idx import *

path_model = './edge_model/BindingDB_tagcnEdge.pt'
path_x = './data/BindingDB/x_np.pkl'
path_edge_index = './data/BindingDB/edge_index_np.pkl'
path_y = './data/BindingDB/y_np.pkl'
model_name = 'tagcnEdge'
save_path = './result/BindingDB_tagcnEdge.json'


def get_model(model_name, num_node_features, num_classes):
    if model_name == 'graphsageEdge':
        model = GraphSAGEdge(num_node_features, num_classes)
    elif model_name == 'tagcnEdge':
        model = TAGCNEdge(num_node_features, num_classes)
    return model


def get_idx_miss_class(target_pre, test_y):
    idx_miss_list = []
    for i in range(len(target_pre)):
        if target_pre[i] != test_y[i]:
            idx_miss_list.append(i)
    idx_miss_list.append(i)
    return idx_miss_list


def get_res_ratio_list(idx_miss_list, select_idx_list, select_ratio_list):
    res_ratio_list = []
    for i in select_ratio_list:
        n = round(len(select_idx_list) * i)
        tmp_select_idx_list = select_idx_list[: n]
        n_hit = len(np.intersect1d(idx_miss_list, tmp_select_idx_list, assume_unique=False, return_indices=False))
        ratio = round(n_hit / len(idx_miss_list), 4)
        res_ratio_list.append(ratio)
    return res_ratio_list


def write_result(content, file_name):
    re = open(file_name, 'a')
    re.write('\n' + content)
    re.close()


def main():
    x = pickle.load(open(path_x, 'rb'))
    edge_index = pickle.load(open(path_edge_index, 'rb'))
    y = pickle.load(open(path_y, 'rb'))

    num_node_features = len(x[0])
    num_classes = len(set(y))

    model = get_model(model_name, num_node_features, num_classes)
    model.load_state_dict(torch.load(path_model))

    idx_np = np.array(list(range(len(y))))
    train_idx, test_idx, train_y, test_y = train_test_split(idx_np, y, test_size=0.2, random_state=17)

    x = torch.from_numpy(x)
    edge_index = torch.from_numpy(edge_index)
    y = torch.from_numpy(y)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.float()
    x = x.to(device).float()
    edge_index = edge_index.to(device)
    y = y.to(device)

    select_ratio_list = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0]
    model.eval()
    with torch.no_grad():
        pre = model(x, edge_index)
    pre = pre.cpu()
    pre_np = pre.numpy()

    pre_np = np.exp(pre_np)
    test_pre_vec = pre_np[test_idx]
    pre_y_test = np.argmax(test_pre_vec, axis=1)
    y_test = y[test_idx]

    deepGini_rank_idx = DeepGini_rank_idx(test_pre_vec)
    lc_rank_idx = LC_rank_idx(test_pre_vec)
    margin_rank_idx = Margin_rank_idx(test_pre_vec)
    entropy_rank_idx = Entropy_rank_idx(test_pre_vec)
    lc_variant_rank_idx = LC_variant_rank_idx(test_pre_vec)
    variance_rank_idx = Variance_rank_idx(test_pre_vec)
    random_rank_idx = Random_rank_idx(test_pre_vec)

    idx_miss_list = get_idx_miss_class(pre_y_test, y_test)

    deepGini_pfd = get_res_ratio_list(idx_miss_list, deepGini_rank_idx, select_ratio_list)
    lc_pfd = get_res_ratio_list(idx_miss_list, lc_rank_idx, select_ratio_list)
    lcvar_pfd = get_res_ratio_list(idx_miss_list, lc_variant_rank_idx, select_ratio_list)
    margin_pfd = get_res_ratio_list(idx_miss_list, margin_rank_idx, select_ratio_list)
    entropy_pfd = get_res_ratio_list(idx_miss_list, entropy_rank_idx, select_ratio_list)
    variance_pfd = get_res_ratio_list(idx_miss_list, variance_rank_idx, select_ratio_list)
    random_pfd = get_res_ratio_list(idx_miss_list, random_rank_idx, select_ratio_list)

    dic = {'deepGini_pfd': deepGini_pfd,
           'lc_pfd': lc_pfd,
           'lcvar_pfd': lcvar_pfd,
           'margin_pfd': margin_pfd,
           'entropy_pfd': entropy_pfd,
           'variance_pfd': variance_pfd,
           'random_pfd': random_pfd,
           }

    json.dump(dic, open(save_path, 'w'), sort_keys=False, indent=4)
    print(dic)


if __name__ == '__main__':
    main()




