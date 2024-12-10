from torch import nn
from torch_geometric import nn as gnn


class GNN(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dims=(), operator=gnn.GCNConv, dropout=0., **kwargs):
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

class GFT_linear_attention(nn.Module):
    def __init__(self, drug_dim, prot_dim, transformer_dim, **kwargs):
        super(GFT_linear_attention, self).__init__()

        self.drug_gnn = GNN(drug_dim, transformer_dim, hidden_dims=[128, 256, 128])

        self.prot_gnn = GNN(prot_dim, transformer_dim, hidden_dims=[128, 64, 128])

        self.attention_layer = None # TODO: fix

    def forward(self, drug, prot):
        x_drug, edge_index_drug, batch_drug = drug.x, drug.edge_index, drug.batch
        x_prot, edge_index_prot, batch_prot = prot.x, prot.edge_index, prot.batch

        embedding_drug = self.drug_gnn(x_drug, edge_index_drug)
        embedding_prot = self.prot_gnn(x_prot, edge_index_prot)

        print(embedding_drug.shape, embedding_prot.shape)
        print(batch_drug.shape, batch_prot.shape)

        # TODO attention

        return
