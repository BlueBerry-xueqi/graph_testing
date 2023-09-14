import pickle
import pickle
import numpy as np
import torch
import argparse
import torch.nn.functional as F
from utils import *
from sklearn.model_selection import train_test_split

ap = argparse.ArgumentParser()
ap.add_argument("--cuda", type=str)
ap.add_argument("--path_x", type=str)
ap.add_argument("--path_edge_index", type=str)
ap.add_argument("--path_y", type=str)
ap.add_argument("--epochs", type=int)
ap.add_argument("--model_name", type=str)
ap.add_argument("--hidden_channel", type=int)
ap.add_argument("--path_save_model", type=str)
args = ap.parse_args()

cuda = args.cuda
path_x = args.path_x
path_edge_index = args.path_edge_index
path_y = args.path_y
epochs = args.epochs
path_save_model = args.path_save_model
model_name = args.model_name
hidden_channel = args.hidden_channel

# python get_target_train.py --cuda 'cuda:1' --path_x '../data/cora/x_np.pkl' --path_edge_index '../data/cora/edge_index_np.pkl' --path_y '../data/cora/y_np.pkl' --epochs 5 --model_name 'gat' --hidden_channel 16 --path_save_model './target_models/cora_gat.pt'


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


def train():

    x = pickle.load(open(path_x, 'rb'))
    edge_index = pickle.load(open(path_edge_index, 'rb'))
    y = pickle.load(open(path_y, 'rb'))

    num_node_features = len(x[0])
    num_classes = len(set(y))
    idx_np = np.array(list(range(len(x))))
    train_all_idx, test_idx, train_all_y, test_y = train_test_split(idx_np, y, test_size=0.2, random_state=17)
    train_idx, candidate_idx, train_y, candidate_y = train_test_split(idx_np, y, test_size=0.5, random_state=17)

    x = torch.from_numpy(x)
    edge_index = torch.from_numpy(edge_index)
    y = torch.from_numpy(y)

    device = torch.device(cuda if torch.cuda.is_available() else 'cpu')
    model = get_model(model_name, num_node_features, hidden_channel, num_classes)
    model = model.to(device)
    x = x.to(device)
    edge_index = edge_index.to(device)
    y = y.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = F.nll_loss(out[train_idx], y[train_idx])
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), path_save_model)

    model.eval()
    with torch.no_grad():
        pred = model(x, edge_index).argmax(dim=1)

        correct = (pred[train_idx] == y[train_idx]).sum()
        acc = int(correct) / len(train_idx)
        print('train:', acc)

        correct = (pred[test_idx] == y[test_idx]).sum()
        acc = int(correct) / len(test_idx)
        print('test:', acc)


if __name__ == '__main__':
    train()

