import pdb

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv  # noqa
import os.path as osp
from torch_geometric.datasets import TUDataset, Planetoid
from torch_geometric.loader import DataLoader
import numpy as np

# load data from TU dataset
path = osp.join(osp.dirname(osp.realpath(__file__)), '../..', 'data', 'TU')
dataset = TUDataset(path, name="DD").shuffle()
n = len(dataset) // 10
index = torch.randperm(len(dataset))
train_index, val_index, test_index, retrain_index = index[:4 * n], index[4 * n:5 * n], index[5 * n: 6 * n], index[
                                                                                                            6 * n:]
# get dataset
train_dataset = dataset[train_index]
val_dataset = dataset[val_index]
test_dataset = dataset[test_index]
retrain_dataset = dataset[retrain_index]

train_loader = DataLoader(train_dataset, batch_size=60, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=60, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=60, shuffle=False)
retrain_loader = DataLoader(retrain_dataset, batch_size=60, shuffle=False)


class Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 16, cached=True,
                             normalize=True)
        self.conv2 = GCNConv(16, out_channels, cached=True,
                             normalize=True)
        # self.conv1 = ChebConv(data.num_features, 16, K=2)
        # self.conv2 = ChebConv(16, data.num_features, K=2)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)


def TU_train(model, optimizer, dataset):
    model.train()
    optimizer.zero_grad()
    for i in range(len(dataset)-1):
        data = dataset[i]
        # x_size = len(model(data))
        # y =[]
        # for i in range(x_size):
        #     y.append(data.y.numpy()[0])
        # y = torch.tensor(y)
        # F.nll_loss(model(data), y).backward()
        data.edge_weight = 0
        model(data)

        optimizer.step()


# @torch.no_grad()
# def TU_test(model, dataset, index):
#     model.eval()
#     logits = model(dataset[0])
#     pred = logits[index].max(1)[1]
#     acc = pred.eq(dataset[0].y[index]).sum().item() / len(index)
#     return acc


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model= Net(dataset.num_features, dataset.num_features).to(device)
optimizer = torch.optim.Adam([
    dict(params=model.conv1.parameters(), weight_decay=5e-4),
    dict(params=model.conv2.parameters(), weight_decay=0)
], lr=0.01)  # Only perform weight-decay on first convolution.

for epoch in range(50):
    TU_train(model, optimizer, retrain_dataset)
    # train_acc = TU_test(model, train_dataset, train_index)
    # val_acc = TU_test(model, val_dataset, val_index)
    # print(f'Epoch: {epoch:03d}, Train: {train_acc:.4f}, '
    #       f'Val: {val_acc:.4f}')
