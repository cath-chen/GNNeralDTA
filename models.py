from torch import nn
from torch_geometric import nn as gnn


class GNN(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dims, operator=gnn.GCNConv, dropout=0., **kwargs):
        super(GNN, self).__init__()

        self.input_layer = operator(in_channels, hidden_dims[0], **kwargs)
        self.hidden_layers = nn.ModuleList(
            [operator(hidden_dims[i], hidden_dims[i + 1], **kwargs) for i in range(len(hidden_dims) - 1)])
        self.output_layer = operator(hidden_dims[-1], out_channels, **kwargs)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index):
        y = self.input_layer(x, edge_index)
        y = self.relu(y)
        y = self.dropout(y)

        for layer in self.hidden_layers:
            y = layer(y, edge_index)
            y = self.relu(y)
            y = self.dropout(y)

        y = self.output_layer(y, edge_index)
        return y

# TODO: Transformer encoder, combination model