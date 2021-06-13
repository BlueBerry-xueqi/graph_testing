import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d as BatchNorm
from torch_geometric.transforms import OneHotDegree
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import GINConv, global_add_pool

from pytorchtools import EarlyStopping,save_model_layer_bame

class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(Net, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for i in range(num_layers):
            mlp = Sequential(
                Linear(in_channels, 2 * hidden_channels),
                BatchNorm(2 * hidden_channels),
                ReLU(),
                Linear(2 * hidden_channels, hidden_channels),
            )
            conv = GINConv(mlp, train_eps=True).jittable()

            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(hidden_channels))

            in_channels = hidden_channels

        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.batch_norm1 = BatchNorm(hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(batch_norm(conv(x, edge_index)))
        x = global_add_pool(x, batch)
        x = F.relu(self.batch_norm1(self.lin1(x)))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)
        
    def compute(self, data):
        return self.forward(data.x, data.edge_index, data.batch)

import os
def train_and_save():
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', '..', 'data', 'TU')
    dataset = TUDataset(path, name='IMDB-BINARY', transform=OneHotDegree(135)) #IMDB-BINARY binary classification
    dataset = dataset.shuffle()
    train_size = int(len(dataset)*0.8)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split( dataset, [train_size, test_size], generator=torch.Generator().manual_seed(0))
    test_loader = DataLoader(test_dataset, batch_size=128)
    train_loader = DataLoader(train_dataset, batch_size=128)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(dataset.num_features, 64*4*2, dataset.num_classes, num_layers=5)
    save_model_layer_bame(model, "saved_model/gin_imdb_binary/layers.json")
    model = model.to(device)
    model = torch.jit.script(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # initialize the early_stopping object
    os.makedirs("saved_model/gin_imdb_binary/", exist_ok=True)
    early_stopping = EarlyStopping(patience=50, verbose=True, path="saved_model/gin_imdb_binary/")

    def train():
        model.train()

        total_loss = 0.
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = F.nll_loss(out, data.y)
            loss.backward()
            total_loss += loss.item() * data.num_graphs
            optimizer.step()
        return total_loss / len(train_dataset)


    @torch.no_grad()
    def test(loader):
        model.eval()

        total_correct = 0
        val_loss = 0
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            loss = F.nll_loss(out, data.y)
            val_loss += loss.item() * data.num_graphs
            pred = out.max(dim=1)[1]
            total_correct += pred.eq(data.y).sum().item()

        return total_correct / len(loader.dataset), val_loss/len(loader.dataset)


    for epoch in range(1, 101):
        loss = train()
        train_acc, train_loss = test(train_loader)
        test_acc, test_loss = test(test_loader)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
            f'Train: {train_acc:.4f}, Test: {test_acc:.4f}')
        early_stopping(test_loss, model, performance={"train_acc":train_acc, "test_acc":test_acc, "train_loss":train_loss, "test_loss":test_loss})
        if early_stopping.early_stop:
            print("Early stopping")
            break
    


if __name__ == "__main__":
    train_and_save()
