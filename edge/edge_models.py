import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, TAGConv


class GraphSAGEdge(torch.nn.Module):
    def __init__(self, features, edge_classes):
        super(GraphSAGEdge, self).__init__()
        self.gat1 = SAGEConv(features, 16)
        self.gat2 = SAGEConv(16, 16)
        self.classifier = torch.nn.Linear(16 * 2, out_features=edge_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.gat1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.gat2(x, edge_index))
        x_src, x_dst = x[edge_index[0]], x[edge_index[1]]
        edge_feat = torch.cat([x_src, x_dst], dim=-1)

        out = self.classifier(edge_feat)

        return F.log_softmax(out, dim=1)


class TAGCNEdge(torch.nn.Module):
    def __init__(self, features, edge_classes):
        super(TAGCNEdge, self).__init__()
        self.gat1 = TAGConv(features, 16)
        self.gat2 = TAGConv(16, 16)
        self.classifier = torch.nn.Linear(16 * 2, edge_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.gat1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.gat2(x, edge_index))
        x_src, x_dst = x[edge_index[0]], x[edge_index[1]]
        edge_feat = torch.cat([x_src, x_dst], dim=-1)

        out = self.classifier(edge_feat)

        return F.log_softmax(out, dim=1)