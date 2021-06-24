import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU
from torch_geometric.nn import GCNConv, global_add_pool
class Net(torch.nn.Module):
    def __init__(self, inputchannel, dim , out_channels):
        super(Net, self).__init__()
        self.conv1 = GCNConv(inputchannel, 16*4, cached=False,
                             normalize=False)
        self.conv2 = GCNConv(16*4, 16*4, cached=False,
                             normalize=False)
        self.conv3 = GCNConv(16*4, dim, cached=False,
                             normalize=False)
        self.lin1 = Linear(dim, 16)
        self.lin2 = Linear(16, out_channels)
      

    def forward(self, x, edge_index, batch ):
        x = F.relu(self.conv1(x, edge_index))
       # x = F.dropout(x, 0.2, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        #x = F.dropout(x, 0.2,training=self.training)
        x = F.relu(self.conv3(x, edge_index))
        #x = F.dropout(x, 0.2,training=self.training)
        x = global_add_pool(x, batch)
        x = self.lin1(x).relu()
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)
       