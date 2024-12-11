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


class LinkAttention(nn.Module):
    def __init__(self, input_dim, n_heads):
        super(LinkAttention, self).__init__()
        self.query = nn.Linear(input_dim, n_heads)
        # self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, masks):
        query = self.query(x).transpose(1, 2)
        value = x

        minus_inf = -9e15 * torch.ones_like(query)
        e = torch.where(masks > 0.5, query, minus_inf)  # (B,heads,seq_len)
        a = self.softmax(e)

        # out = torch.matmul(a, value).view(query.shape[0], -1)
        out = torch.matmul(a, value)
        out = torch.sum(out, dim=1).squeeze()
        return out, a


class linear_attention(nn.Module):
    def __init__(self, input_dim, n_heads):
        super(linear_attention, self).__init__()
        self.drug_attention = LinkAttention(input_dim, n_heads)
        self.prot_attention = LinkAttention(input_dim, n_heads)
        self.comb_attention = LinkAttention(input_dim, n_heads)

    def forward(self, drug, prot, mask_drug, mask_prot):
        comb = torch.cat((drug, prot), dim=1)
        mask_comb = torch.cat((mask_drug, mask_prot), dim=1)

        drug_attention = self.drug_attention(drug, mask_drug)[0]
        prot_attention = self.prot_attention(prot, mask_prot)[0]
        comb_attention = self.comb_attention(comb, mask_comb)[0]

        attention = torch.cat((drug_attention, prot_attention, comb_attention), dim=1)

        return attention


class cross_attention(nn.Module):
    def __init__(self, dim, aggregation='max'):
        super(cross_attention, self).__init__()
        self.attention_1 = nn.MultiheadAttention(dim, dim, batch_first=True)
        self.attention_2 = nn.MultiheadAttention(dim, dim, batch_first=True)
        self.aggregation = aggregation

    def forward(self, drug, prot, mask_drug=None, mask_prot=None):
        attention_1 = self.attention_1(drug, prot, prot, key_padding_mask=~mask_prot)[0]
        attention_2 = self.attention_2(prot, drug, drug, key_padding_mask=~mask_drug)[0]

        if self.aggregation == 'mean':
            attention_1 = torch.mean(attention_1, dim=1)
            attention_2 = torch.mean(attention_2, dim=1)
        elif self.aggregation == 'max':
            attention_1 = torch.max(attention_1, dim=1)[0]
            attention_2 = torch.max(attention_2, dim=1)[0]

        attention = torch.concat((attention_1, attention_2), dim=1)

        return attention


class AttentionGNNeral(nn.Module):
    def __init__(self, drug_dim, prot_dim, attention_dim, attention='cross', gnn_dropout=0., fnn_dropout=0.,
                 time=False):
        super(AttentionGNNeral, self).__init__()

        self.time = time

        self.drug_gnn = GNN(drug_dim, attention_dim, hidden_dims=[64] * 0, dropout=gnn_dropout)
        self.prot_gnn = GNN(prot_dim, attention_dim, hidden_dims=[64] * 0, dropout=gnn_dropout)

        assert attention in ['cross', 'linear']
        if attention == 'cross':
            self.attention = cross_attention(attention_dim)
            num_embeddings = 2
        elif attention == 'linear':
            self.attention = linear_attention(attention_dim, n_heads=1)
            num_embeddings = 3

        self.classifier = nn.Sequential(
            nn.Linear(attention_dim * num_embeddings, 1024),
            nn.ReLU(),
            nn.Dropout(fnn_dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(fnn_dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(fnn_dropout),
            nn.Linear(256, 1),
        )

    def forward(self, drug, prot):
        x_drug, edge_index_drug, batch_drug = drug.x, drug.edge_index, drug.batch
        x_prot, edge_index_prot, batch_prot = prot.x, prot.edge_index, prot.batch

        if self.time:
            start = time.time()

        embedding_drug = self.drug_gnn(x_drug, edge_index_drug)
        embedding_prot = self.prot_gnn(x_prot, edge_index_prot)

        if self.time:
            print(f"gnn {time.time() - start:.2f}s")
            start = time.time()
        embedding_drug, mask_drug = to_dense_batch(embedding_drug, batch_drug)
        embedding_prot, mask_prot = to_dense_batch(embedding_prot, batch_prot, max_num_nodes=1000)

        attention = self.attention(embedding_drug, embedding_prot, mask_drug, mask_prot)

        if self.time:
            print(f"attention {time.time() - start:.2f}s")
            start = time.time()

        output = self.classifier(attention)
        if self.time:
            print(f"classifier {time.time() - start:.2f}s")

        return output
