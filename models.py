import time

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


class cross_attention(nn.Module):
    def __init__(self, dim, aggregation='max'):
        super(cross_attention, self).__init__()
        self.attention_1 = nn.MultiheadAttention(dim, dim, batch_first=True)
        self.attention_2 = nn.MultiheadAttention(dim, dim, batch_first=True)
        self.aggregation = aggregation

    def forward(self, drug, prot, mask_drug=None, mask_prot=None):
        attention_1 = self.attention_1(drug, prot, prot)[0]
        attention_2 = self.attention_2(prot, drug, drug)[0]

        if self.aggregation == 'mean':
            attention_1 = torch.mean(attention_1, dim=1)
            attention_2 = torch.mean(attention_2, dim=1)
        elif self.aggregation == 'max':
            attention_1 = torch.max(attention_1, dim=1)[0]
            attention_2 = torch.max(attention_2, dim=1)[0]

        attention = torch.concat((attention_1, attention_2), dim=1)

        return attention


class AttentionGNNeral(nn.Module):
    def __init__(self, drug_dim, prot_dim, transformer_dim, attention='cross', gnn_dropout=0., fnn_dropout=0.):
        super(AttentionGNNeral, self).__init__()

        self.drug_gnn = GNN(drug_dim, transformer_dim, hidden_dims=[128] * 4, dropout=gnn_dropout)
        self.prot_gnn = GNN(prot_dim, transformer_dim, hidden_dims=[64] * 2, dropout=gnn_dropout)

        assert attention in ['cross', 'linear']
        if attention == 'cross':
            self.attention = cross_attention(transformer_dim)
            num_embeddings = 2
        elif attention == 'linear':
            self.attention = linear_attention(transformer_dim)
            num_embeddings = 3

        self.classifier = nn.Sequential(
            nn.Linear(transformer_dim * num_embeddings, 1024),
            nn.ReLU(),
            nn.Dropout(fnn_dropout),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(fnn_dropout),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(fnn_dropout),
            nn.Linear(256, 1),
        )

    def forward(self, drug, prot):
        x_drug, edge_index_drug, batch_drug = drug.x, drug.edge_index, drug.batch
        x_prot, edge_index_prot, batch_prot = prot.x, prot.edge_index, prot.batch

        start = time.time()
        embedding_drug = self.drug_gnn(x_drug, edge_index_drug)
        embedding_prot = self.prot_gnn(x_prot, edge_index_prot)
        print(f"gnn {time.time() - start:.2f}s")
        start = time.time()
        embedding_drug, mask_drug = to_dense_batch(embedding_drug, batch_drug)
        embedding_prot, mask_prot = to_dense_batch(embedding_prot, batch_prot, max_num_nodes=1000)

        attention = self.attention(embedding_drug, embedding_prot, mask_drug, mask_prot)
        print(f"attention {time.time() - start:.2f}s")
        start = time.time()

        output = self.classifier(attention)
        output = torch.squeeze(output)
        print(f"classifier {time.time() - start:.2f}s")

        return output
