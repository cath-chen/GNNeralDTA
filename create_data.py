import json
import os
import pickle
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch_geometric import data as DATA
from torch_geometric.data import Batch

from graph_conversion import cif_to_graph, smile_to_graph


def collate(data_list):
    batchA = Batch.from_data_list([data[0] for data in data_list])
    batchB = Batch.from_data_list([data[1] for data in data_list])
    batchC = torch.tensor([data[2] for data in data_list], dtype=torch.float32)
    return batchA, batchB, batchC


class GraphPairDataset(Dataset):
    def __init__(self, pairs, drug_graphs, prot_graphs):
        self.pairs = pairs
        self.drug_graphs = drug_graphs
        self.prot_graphs = prot_graphs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        drug, prot, affinity = self.pairs[idx]
        drug = self.drug_graphs[drug]
        prot = self.prot_graphs[prot]

        return drug, prot, affinity


def create_dataloader(batch_size=64):
    all_prots = []
    print('convert data from DeepDTA for davis')
    fpath = 'data/davis/'
    train_fold = json.load(open(fpath + "folds/train_fold_setting1.txt"))
    train_fold = [ee for e in train_fold for ee in e]
    test_fold = json.load(open(fpath + "folds/test_fold_setting1.txt"))
    ligands = json.load(open(fpath + "ligands_can.txt"), object_pairs_hook=OrderedDict)
    proteins = json.load(open(fpath + "proteins.txt"), object_pairs_hook=OrderedDict)
    affinity = pickle.load(open(fpath + "Y", "rb"), encoding='latin1')

    drugs = list(ligands.keys())
    prots = list(proteins.keys())
    affinity = [-np.log10(y / 1e9) for y in affinity]
    affinity = np.asarray(affinity)

    opts = ['train', 'test']
    for opt in opts:
        rows, cols = np.where(np.isnan(affinity) == False)
        if opt == 'train':
            rows, cols = rows[train_fold], cols[train_fold]
        elif opt == 'test':
            rows, cols = rows[test_fold], cols[test_fold]

        data = []
        for pair_ind in range(len(rows)):
            data.append((drugs[rows[pair_ind]], prots[cols[pair_ind]], affinity[rows[pair_ind], cols[pair_ind]]))

        if opt == 'train':
            train_data = data
        elif opt == 'test':
            test_data = data

    if os.path.isfile('data/drugs.pkl'):
        with open('data/drugs.pkl', 'rb') as f:
            drugs = pickle.load(f)
        print(f"loaded {len(drugs)} drugs")
    else:
        drugs = {drug: smile_to_graph(smile) for drug, smile in ligands.items()}
        with open('data/drugs.pkl', 'wb') as f:
            pickle.dump(drugs, f)
        print(f"converted {len(drugs)} drugs")

    if os.path.isfile('data/prots.pkl'):
        with open('data/prots.pkl', 'rb') as f:
            prots = pickle.load(f)
        print(f"loaded {len(prots)} prots")
    else:
        prots = {}
        with ThreadPoolExecutor() as executor:
            future_to_key = {executor.submit(cif_to_graph, f"pdb_files/{prot}.cif"): prot for prot in proteins.keys()}
            for future in as_completed(future_to_key):
                key = future_to_key[future]
                prots[key] = future.result()
        with open('data/prots.pkl', 'wb') as f:
            pickle.dump(prots, f)
        print(f"converted {len(prots)} prots")

    drug_graphs = {key: DATA.Data(x=torch.Tensor(np.array(features)),
                                  edge_index=torch.LongTensor(np.array(edge_index)).transpose(1, 0)) for
                   key, (_, features, edge_index) in drugs.items()}
    prot_graphs = {key: DATA.Data(x=torch.Tensor(np.array(features)),
                                  edge_index=torch.LongTensor(np.array(edge_index)).transpose(1, 0),
                                  edge_attr=torch.LongTensor(np.array(edge_weights))) for
                   key, (_, features, edge_index, edge_weights) in prots.items()}

    train_dataset = GraphPairDataset(train_data, drug_graphs, prot_graphs)
    test_dataset = GraphPairDataset(test_data, drug_graphs, prot_graphs)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)

    return train_loader, test_loader


if __name__ == '__main__':
    train_loader, test_loader = create_dataloader()
