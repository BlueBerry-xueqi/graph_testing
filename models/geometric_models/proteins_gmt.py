import os.path as osp

import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphMultisetTransformer


class Net(torch.nn.Module):
    def __init__(self, dataset):
        super().__init__()
        avg_num_nodes = int(dataset.data.num_nodes / len(dataset))

        self.conv1 = GCNConv(dataset.num_features, 32)
        self.conv2 = GCNConv(32, 32)
        self.conv3 = GCNConv(32, 32)

        self.pool = GraphMultisetTransformer(
            in_channels=96,
            hidden_channels=64,
            out_channels=32,
            Conv=GCNConv,
            num_nodes=avg_num_nodes,
            pooling_ratio=0.25,
            pool_sequences=['GMPool_G', 'SelfAtt', 'GMPool_I'],
            num_heads=4,
            layer_norm=False,
        )

        self.lin1 = Linear(32, 16)
        self.lin2 = Linear(16, dataset.num_classes)

    def forward(self, data):
        x0, edge_index, batch = data.x, data.edge_index, data.batch
        x1 = self.conv1(x0, edge_index).relu()
        x2 = self.conv2(x1, edge_index).relu()
        x3 = self.conv3(x2, edge_index).relu()
        x = torch.cat([x1, x2, x3], dim=-1)

        x = self.pool(x, batch, edge_index)

        x = self.lin1(x).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        return x.log_softmax(dim=-1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(model, train_loader, train_dataset, optimizer):
    model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        total_loss += data.num_graphs * float(loss)
        optimizer.step()
    return total_loss / len(train_dataset)


@torch.no_grad()
def test(loader, model):
    model.eval()

    total_correct = 0
    for data in loader:
        data = data.to(device)
        out = model(data)
        total_correct += int((out.argmax(dim=-1) == data.y).sum())
    return total_correct / len(loader.dataset)
