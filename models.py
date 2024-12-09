from torch import nn
from torch_geometric import nn as gnn


class GNN(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dims=[], operator=gnn.GCNConv, dropout=0., **kwargs):
        super(GNN, self).__init__()

        layer_dims = [in_channels] + hidden_dims + [out_channels]
        self.layers = nn.ModuleList([
            operator(layer_dims[i], layer_dims[i + 1], **kwargs) for i in range(len(layer_dims) - 1)
        ])

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index):
        for layer in self.layers:
            x = layer(x, edge_index)
            x = self.relu(x)
            x = self.dropout(x)

        return x

# TODO: Transformer encoder, combination model
