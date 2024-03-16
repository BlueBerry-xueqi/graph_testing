import torch
import pickle
import json
import random
import numpy as np
from cluster_metrics.GMM import GMM_metrics
from cluster_metrics.Hierarchical import Hierarchical_metrics
from cluster_metrics.Kmeans import KMeans_metrics, KMeans_plusplus_metrics
from cluster_metrics.MiniBatchKMeans import MiniBatchKMeans_metrics
from sklearn.model_selection import train_test_split
from edge_models import GraphSAGEdge, TAGCNEdge
from sklearn.metrics import accuracy_score

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--path_model", type=str)
ap.add_argument("--path_x", type=str)
ap.add_argument("--path_edge_index", type=str)
ap.add_argument("--path_y", type=str)
ap.add_argument("--model_name", type=str)
ap.add_argument("--save_path", type=str)
ap.add_argument("--epochs", type=int)
args = ap.parse_args()

path_model = args.path_model
path_x = args.path_x
path_edge_index = args.path_edge_index
path_y = args.path_y
model_name = args.model_name
save_path = args.save_path
epochs = args.epochs

# path_model = './edge_model/BindingDB_tagcnEdge.pt'
# path_x = './data/BindingDB/x_np.pkl'
# path_edge_index = './data/BindingDB/edge_index_np.pkl'
# path_y = './data/BindingDB/y_np.pkl'
# model_name = 'tagcnEdge'
# save_path = './result/estimation_BindingDB_tagcnEdge.json'
# epochs = 5


def write_result(content, file_name):
    re = open(file_name, 'a')
    re.write('\n' + content)
    re.close()


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

    model.eval()
    with torch.no_grad():
        pre = model(x, edge_index)
    pre = pre.cpu()
    pre_np = pre.numpy()

    pre_np = np.exp(pre_np)

    test_pre_vec = pre_np[test_idx]
    pre_y_test = np.argmax(test_pre_vec, axis=1)
    y_test = y[test_idx]

    select_n_list = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
    dic = {'GMM': [],
           'Hierarchical': [],
           'KMeans': [],
           'KMeansPlus': [],
           'MiniBatchKMeans': [],
           'Random':[]
           }

    acc = accuracy_score(pre_y_test, y_test)

    for select_n in select_n_list:
        GMM_diff = 0
        Hierarchical_diff = 0
        KMeans_diff = 0
        KMeansPlus_diff = 0
        MiniBatchKMeans_diff = 0
        Random_diff = 0
        for _ in range(epochs):

            idx_list = KMeans_metrics(test_pre_vec, select_n)
            KMeans_diff += np.power((accuracy_score(pre_y_test[idx_list], y_test[idx_list])-acc), 2)
            print('====KMeans===', _)

            idx_list = KMeans_plusplus_metrics(test_pre_vec, select_n)
            KMeansPlus_diff += np.power((accuracy_score(pre_y_test[idx_list], y_test[idx_list])-acc), 2)
            print('====KMeansPlus===', _)

            idx_list = MiniBatchKMeans_metrics(test_pre_vec, select_n)
            MiniBatchKMeans_diff += np.power((accuracy_score(pre_y_test[idx_list], y_test[idx_list])-acc), 2)
            print('====MiniBatchKMeans===', _)

            # idx_list = Hierarchical_metrics(test_pre_vec, select_n)
            # Hierarchical_diff += np.power((accuracy_score(pre_y_test[idx_list], y_test[idx_list])-acc), 2)
            # print('====Hierarchical===', _)

            # idx_list = GMM_metrics(test_pre_vec, select_n)
            # GMM_diff += np.power((np.power((accuracy_score(pre_y_test[idx_list], y_test[idx_list])-acc), 2)-acc), 2)
            # print('====GMM===', _)


        dic['KMeans'].append(np.sqrt(KMeans_diff / epochs))
        dic['KMeansPlus'].append(np.sqrt(KMeansPlus_diff / epochs))
        dic['MiniBatchKMeans'].append(np.sqrt(MiniBatchKMeans_diff / epochs))

        # dic['Hierarchical'].append(np.sqrt(Hierarchical_diff / epochs))
        # dic['GMM'].append(np.sqrt(GMM_diff / epochs))

        for i in range(1000):
            idx_list = random.sample(range(len(test_pre_vec)), select_n)
            Random_diff += np.power((accuracy_score(pre_y_test[idx_list], y_test[idx_list]) - acc), 2)
        dic['Random'].append(np.sqrt(Random_diff / 1000))

        print(dic)

    json.dump(dic, open(save_path, 'w'), sort_keys=True, indent=4)


if __name__ == '__main__':
    main()
