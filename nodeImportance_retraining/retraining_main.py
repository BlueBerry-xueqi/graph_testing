import torch.nn.functional as F
from importance_rank_idx import *
from utils import *
import pickle
import networkx as nx
import json
import torch

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--cuda", type=str)
ap.add_argument("--model_name", type=str)
ap.add_argument("--target_model_path", type=str)
ap.add_argument("--path_x", type=str)
ap.add_argument("--path_edge_index", type=str)
ap.add_argument("--path_y", type=str)
ap.add_argument("--path_save", type=str)

args = ap.parse_args()
cuda = args.cuda
model_name = args.model_name
target_model_path = args.target_model_path
path_x = args.path_x
path_edge_index = args.path_edge_index
path_y = args.path_y
path_save = args.path_save

# python retraining_main.py --cuda 'cuda:1' --model_name 'gcn' --target_model_path './target_models/cora_gcn.pt' --path_x '../data/cora/x_np.pkl' --path_edge_index '../data/cora/edge_index_np.pkl' --path_y '../data/cora/y_np.pkl' --path_save './result/cora_gcn.json'


target_hidden_channel = 16
hidden_channel = 16

epochs = 5
ratio = 0.1

num_node_features, num_classes, x, edge_index, y, test_y, train_all_idx, test_idx = load_data(path_x, path_edge_index,
                                                                                              path_y)

train_idx, candidate_idx, train_y, candidate_y = train_test_split(train_all_idx, y[train_all_idx], test_size=0.5,
                                                                  random_state=17)

target_model = load_target_model(model_name, num_node_features, target_hidden_channel, num_classes, target_model_path)

device = torch.device(cuda if torch.cuda.is_available() else 'cpu')


def get_graph(path_edge_index_np):
    G = nx.Graph()
    edge_index_np = pickle.load(open(path_edge_index_np, 'rb'))

    left_point = edge_index_np[0]
    right_point = edge_index_np[1]
    for i in range(len(left_point)):
        G.add_edge(left_point[i], right_point[i])

    while not nx.is_connected(G):
        components = list(nx.connected_components(G))
        component1 = random.choice(components)
        components.remove(component1)
        component2 = random.choice(components)
        node1 = random.choice(list(component1))
        node2 = random.choice(list(component2))
        G.add_edge(node1, node2)

    return G


def get_graph_degree(path_edge_index_np):
    G = nx.Graph()
    edge_index_np = pickle.load(open(path_edge_index_np, 'rb'))

    left_point = edge_index_np[0]
    right_point = edge_index_np[1]
    for i in range(len(left_point)):
        G.add_edge(left_point[i], right_point[i])

    return G


def get_model(model_name, num_node_features, hidden_channel, num_classes):
    if model_name == 'gcn':
        model = Target_GCN(num_node_features, hidden_channel, num_classes)
    elif model_name == 'gat':
        model = Target_GAT(num_node_features, hidden_channel, num_classes)
    elif model_name == 'graphsage':
        model = Target_GraphSAGE(num_node_features, hidden_channel, num_classes)
    elif model_name == 'tagcn':
        model = Target_TAGCN(num_node_features, hidden_channel, num_classes)
    return model


def get_retrain(rank_list, x, y, edge_index):

    x = x.to(device)
    edge_index = edge_index.to(device)
    y = y.to(device)

    acc_list = []
    for s in range(10):
        model = get_model(model_name, num_node_features, hidden_channel, num_classes)
        model.load_state_dict(torch.load(target_model_path))
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

        model.train()
        tmp_idx = rank_list[:int(len(rank_list)*ratio)]
        select_idx = list(train_idx)+list(tmp_idx)
        for _ in range(epochs):
            optimizer.zero_grad()
            out = model(x, edge_index)
            loss = F.nll_loss(out[select_idx], y[select_idx])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            pred = model(x, edge_index).argmax(dim=1)
            correct = (pred[test_idx] == y[test_idx]).sum()
            acc = int(correct) / len(test_idx)
            acc_list.append(acc)
        print('======================', s, '======================')

    acc = sum(acc_list) / len(acc_list)
    return acc


def main():

    G = get_graph(path_edge_index)
    G_degree = get_graph_degree(path_edge_index)

    x_candidate_target_model_pre = target_model(x, edge_index).detach().numpy()[candidate_idx]

    random_rank_idx = Random_rank_idx(x_candidate_target_model_pre)

    degree_rank_idx = Degree_rank_idx(G_degree, candidate_idx)
    eccentricity_rank_idx = Eccentricity_rank_idx(G, candidate_idx)
    center_rank_idx = Center_rank_idx(G, candidate_idx)
    bc_rank_idx = Betweenness_Centrality_rank_idx(G, candidate_idx)
    ec_rank_idx = Eigenvector_Centrality_rank_idx(G, candidate_idx)
    pr_rank_idx = PageRank_rank_idx(G_degree, candidate_idx)
    hits_rank_idx = Hits_rank_idx(G_degree, candidate_idx)

    random_value = get_retrain(random_rank_idx, x, y, edge_index)

    degree_value = get_retrain(degree_rank_idx, x, y, edge_index)
    eccentricity_value = get_retrain(eccentricity_rank_idx, x, y, edge_index)
    center_value = get_retrain(center_rank_idx, x, y, edge_index)
    bc_value = get_retrain(bc_rank_idx, x, y, edge_index)
    ec_value = get_retrain(ec_rank_idx, x, y, edge_index)
    pr_value = get_retrain(pr_rank_idx, x, y, edge_index)
    hits_value = get_retrain(hits_rank_idx, x, y, edge_index)

    dic = {
        'Random': random_value,
        'Degree': degree_value,
        'Eccentricity': eccentricity_value,
        'Center': center_value,
        'BC': bc_value,
        'EC': ec_value,
        'PageRank': pr_value,
        'Hits': hits_value,
    }
    print(dic)
    json.dump(dic, open(path_save, 'w'), sort_keys=True, indent=4)


if __name__ == '__main__':
    main()


