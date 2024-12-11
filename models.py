import torch
from torch import nn
from torch_geometric import nn as gnn
from torch_geometric.utils import to_dense_batch


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

class linear_attention(nn.Module):
    def __init__(self, dim, **kwargs):
        super(linear_attention, self).__init__()

        self.Q = nn.Linear(dim, dim)
        self.K = nn.Linear(dim, dim)
        self.V = nn.Linear(dim, dim)

        self.attention = nn.MultiheadAttention(dim, dim)

    def forward(self, x):

        q = self.Q(x)
        k = self.K(x)
        v = self.V(x)

        out = self.attention(q, k, v)

        return out

class GFT_linear_attention(nn.Module):
    def __init__(self, drug_dim, prot_dim, transformer_dim, **kwargs):
        super(GFT_linear_attention, self).__init__()

        self.drug_gnn = GNN(drug_dim, transformer_dim, hidden_dims=[128, 128, 128])
        self.prot_gnn = GNN(prot_dim, transformer_dim, hidden_dims=[128, 128, 128])

        self.attention_drug = linear_attention(transformer_dim)
        self.attention_prot = linear_attention(transformer_dim)
        self.attention_combined = linear_attention(transformer_dim)

        self.attention_layer = None # TODO: fix

    def forward(self, drug, prot):
        x_drug, edge_index_drug, batch_drug = drug.x, drug.edge_index, drug.batch
        x_prot, edge_index_prot, batch_prot = prot.x, prot.edge_index, prot.batch

        embedding_drug = self.drug_gnn(x_drug, edge_index_drug)
        embedding_prot = self.prot_gnn(x_prot, edge_index_prot)

        print(embedding_drug.shape, embedding_prot.shape)
        print(batch_drug.shape, batch_prot.shape)

        embedding_drug = to_dense_batch(embedding_drug, batch_drug)[0]
        embedding_prot = to_dense_batch(embedding_prot, batch_prot, max_num_nodes=1000)[0]
        embedding_combined = torch.concat((embedding_drug, embedding_prot), dim=1)

        print(embedding_drug.shape, embedding_prot.shape, embedding_combined.shape)
        print(batch_drug.shape, batch_prot.shape)

        embedding_drug = self.attention_drug(embedding_drug)
        embedding_prot = self.attention_prot(embedding_prot)
        embedding_combined = self.attention_combined(embedding_combined)

        print(len(embedding_drug), len(embedding_prot), len(embedding_combined))

        # TODO attention

        return
