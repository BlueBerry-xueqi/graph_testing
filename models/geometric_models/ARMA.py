import torch
import torch.nn.functional as F
from torch_geometric.nn import ARMAConv


class Net(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()

        self.conv1 = ARMAConv(num_features, 16, num_stacks=3,
                              num_layers=2, shared_weights=True, dropout=0.25)

        self.conv2 = ARMAConv(16, num_classes, num_stacks=3,
                              num_layers=2, shared_weights=True, dropout=0.25,
                              act=lambda x: x)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
