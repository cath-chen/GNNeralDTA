import torch
from torch import nn
from torch_geometric import nn as gnn
from torch_geometric.utils import to_dense_batch


class GNN(nn.Module):
    """
    Flexible GNN block that works with multitude of convolution (message passing) functions.
    """

    def __init__(self, in_channels, out_channels, hidden_dims=(), conv=gnn.GINConv, dropout=0.):
        super(GNN, self).__init__()

        layer_dims = [in_channels] + hidden_dims + [out_channels]
        self.layers = nn.ModuleList([
            conv(layer_dims[i], layer_dims[i + 1]) for i in range(len(layer_dims) - 1)
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
    """
    Linear attention layer from FusionDTA to compute linear self attention.
    """

    def __init__(self, input_dim, n_heads):
        super(LinkAttention, self).__init__()
        self.query = nn.Linear(input_dim, n_heads)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, masks):
        query = self.query(x).transpose(1, 2)
        value = x

        minus_inf = -9e15 * torch.ones_like(query)
        e = torch.where(masks > 0.5, query, minus_inf)  # (B,heads,seq_len)
        a = self.softmax(e)

        out = torch.matmul(a, value)
        out = torch.sum(out, dim=1).squeeze()
        return out, a


class LinearAttention(nn.Module):
    """
    Linear attention block computing the linear attention of drugs, proteins, and their concatenation.
    """

    def __init__(self, input_dim, n_heads):
        super(LinearAttention, self).__init__()
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


class FullCrossAttention(nn.Module):
    """
    Computes full cross attention from drugs to proteins and proteins to drugs.
    """

    def __init__(self, dim, n_heads=1):
        super(FullCrossAttention, self).__init__()
        self.attention_1 = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.attention_2 = nn.MultiheadAttention(dim, n_heads, batch_first=True)

    def forward(self, drug, prot, mask_drug=None, mask_prot=None):
        attention_1 = self.attention_1(drug, prot, prot, key_padding_mask=~mask_prot)[0]
        attention_2 = self.attention_2(prot, drug, drug, key_padding_mask=~mask_drug)[0]

        attention_1 = torch.max(attention_1, dim=1)[0]
        attention_2 = torch.max(attention_2, dim=1)[0]

        attention = torch.concat((attention_1, attention_2), dim=1)

        return attention

# {'learn_rate': 0.001, 'n_heads': 4, 'attention_dim': 256, 'conv': <class 'torch_geometric.nn.conv.graph_conv.GraphConv'>, 'prot_gnn_layers': 2, 'drug_gnn_layers': 5, 'gnn_dropout': 0.1, 'fnn_dropout': 0.2}

class AttentionGNNeral(nn.Module):
    """
    DTA prediction model employing GNNs for both drug and protein embeddings.
    These are compared with an attention mechanism and the final output is predicted with a MLP.
    """

    def __init__(self, drug_dim, prot_dim, attention_dim=256, attention='linear', drug_gnn_layers=5, prot_gnn_layers=2,
                 gnn_dimension=128, conv=gnn.GraphConv, gnn_dropout=0.1, fnn_dropout=0.2, n_heads=1, **kwargs):
        super(AttentionGNNeral, self).__init__()

        self.drug_gnn = GNN(drug_dim, attention_dim, hidden_dims=[gnn_dimension] * (drug_gnn_layers - 1), conv=conv,
                            dropout=gnn_dropout)
        self.prot_gnn = GNN(prot_dim, attention_dim, hidden_dims=[gnn_dimension] * (prot_gnn_layers - 1), conv=conv,
                            dropout=gnn_dropout)

        assert attention in ['cross', 'linear']
        if attention == 'cross':
            self.attention = FullCrossAttention(attention_dim, n_heads=n_heads)
            num_embeddings = 2
        elif attention == 'linear':
            self.attention = LinearAttention(attention_dim, n_heads=1)
            num_embeddings = 3

        self.classifier = nn.Sequential(
            nn.Linear(attention_dim * num_embeddings, 1024),
            nn.ReLU(),
            nn.Dropout(fnn_dropout),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(fnn_dropout),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(fnn_dropout),
            nn.Linear(256, 1)
        )

    def forward(self, drug, prot):
        x_drug, edge_index_drug, batch_drug = drug.x, drug.edge_index, drug.batch
        x_prot, edge_index_prot, batch_prot = prot.x, prot.edge_index, prot.batch

        embedding_drug = self.drug_gnn(x_drug, edge_index_drug)
        embedding_prot = self.prot_gnn(x_prot, edge_index_prot)

        embedding_drug, mask_drug = to_dense_batch(embedding_drug, batch_drug)
        embedding_prot, mask_prot = to_dense_batch(embedding_prot, batch_prot, max_num_nodes=1000)

        attention = self.attention(embedding_drug, embedding_prot, mask_drug, mask_prot)

        output = self.classifier(attention)

        return output
