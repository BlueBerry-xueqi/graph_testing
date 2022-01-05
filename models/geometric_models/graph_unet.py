import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphUNet
from torch_geometric.utils import dropout_adj


class Net(torch.nn.Module):
    def __init__(self, dataset):
        super().__init__()
        data = dataset[0]
        pool_ratios = [2000 / data.num_nodes, 0.5]
        self.unet = GraphUNet(dataset.num_features, 32, dataset.num_classes,
                              depth=3, pool_ratios=pool_ratios)

    def forward(self, data):
        edge_index, _ = dropout_adj(data.edge_index, p=0.2,
                                    force_undirected=True,
                                    num_nodes=data.num_nodes,
                                    training=self.training)
        x = F.dropout(data.x, p=0.92, training=self.training)

        x = self.unet(x, edge_index)
        return F.log_softmax(x, dim=1)


