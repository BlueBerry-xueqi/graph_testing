import pickle
import pickle
import numpy as np
import torch
import argparse
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from edge_models import GraphSAGEdge, TAGCNEdge

path_x = './data/DrugBank/x_np.pkl'
path_edge_index = './data/DrugBank/edge_index_np.pkl'
path_y = './data/DrugBank/y_np.pkl'
model_name = 'tagcnEdge'
epochs = 600
path_save_model = './edge_model/DrugBank_tagcnEdge.pt'


def get_model(model_name, num_node_features, num_classes):

    if model_name == 'graphsageEdge':
        model = GraphSAGEdge(num_node_features, num_classes)
    elif model_name == 'tagcnEdge':
        model = TAGCNEdge(num_node_features, num_classes)
    return model


def train():

    x = pickle.load(open(path_x, 'rb'))
    edge_index = pickle.load(open(path_edge_index, 'rb'))
    y = pickle.load(open(path_y, 'rb'))

    num_node_features = len(x[0])
    num_classes = len(set(y))

    idx_np = np.array(list(range(len(y))))
    train_idx, test_idx, train_y, test_y = train_test_split(idx_np, y, test_size=0.2, random_state=17)

    x = torch.from_numpy(x)
    edge_index = torch.from_numpy(edge_index)
    y = torch.from_numpy(y)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(model_name, num_node_features, num_classes)
    model = model.to(device)
    model.float()  # Convert model parameters to float
    x = x.to(device).float()  # Convert input data to float
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
        print('========',epoch,'=========')

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

