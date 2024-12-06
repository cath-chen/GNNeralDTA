import copy
import os
from collections import OrderedDict

import numpy as np
import pandas as pd
import pickle
import sys
import json
import time
import torch
import torch.nn as nn
from rdkit import Chem
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch_geometric import data as DATA
from torch_geometric.data import Batch

import sys

from graph_conversion import cif_to_graph, smile_to_graph


def collate(data_list):
    batchA = Batch.from_data_list([data[0] for data in data_list])
    batchB = Batch.from_data_list([data[1] for data in data_list])
    return batchA, batchB


class GraphPairDataset(Dataset):
    def __init__(self, pairs, smile_graphs, prot_graphs):
        self.pairs = pairs
        self.smile_graphs = smile_graphs
        self.prot_graphs = prot_graphs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        smile, prot = self.pairs[idx]
        GCNData_Smile = self.smile_graphs[smile]
        GCNData_Prot = self.prot_graphs[prot]
        return GCNData_Smile, GCNData_Prot


# from DeepDTA data
all_prots = []
print('convert data from DeepDTA for davis')
fpath = 'data/davis/'
train_fold = json.load(open(fpath + "folds/train_fold_setting1.txt"))
train_fold = [ee for e in train_fold for ee in e]
valid_fold = json.load(open(fpath + "folds/test_fold_setting1.txt"))
ligands = json.load(open(fpath + "ligands_can.txt"), object_pairs_hook=OrderedDict)
proteins = json.load(open(fpath + "proteins.txt"), object_pairs_hook=OrderedDict)
affinity = pickle.load(open(fpath + "Y", "rb"), encoding='latin1')
drugs = {drug: smile_to_graph(smile) for drug, smile in ligands.items()}
print(f"converted {len(drugs)} drugs")
prots = {prot: cif_to_graph(f"pdb_files/{prot}.cif") for prot in proteins.keys()}
print(f"converted {len(prots)} drugs")
sys.exit()
for d in ligands.keys():
    lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]), isomericSmiles=True)
    drugs.append(lg)
for t in proteins.keys():
    prots.append(t)
affinity = [-np.log10(y / 1e9) for y in affinity]
affinity = np.asarray(affinity)
opts = ['train', 'test']
for opt in opts:
    rows, cols = np.where(np.isnan(affinity) == False)
    if opt == 'train':
        rows, cols = rows[train_fold], cols[train_fold]
    elif opt == 'test':
        rows, cols = rows[valid_fold], cols[valid_fold]
    with open('data/davis_' + opt + '.csv', 'w') as f:
        f.write('compound_iso_smiles,target_sequence,affinity\n')
        for pair_ind in range(len(rows)):
            ls = []
            ls += [drugs[rows[pair_ind]]]
            ls += [prots[cols[pair_ind]]]
            ls += [affinity[rows[pair_ind], cols[pair_ind]]]
            f.write(','.join(map(str, ls)) + '\n')
print('\ndataset: davis')
print('train_fold:', len(train_fold))
print('test_fold:', len(valid_fold))
print('len(set(drugs)),len(set(prots)):', len(set(drugs)), len(set(prots)))
all_prots += list(set(prots))
sys.exit()

compound_iso_smiles = []
opts = ['train', 'test']
for opt in opts:
    df = pd.read_csv('data/davis_' + opt + '.csv')
    compound_iso_smiles += list(df['compound_iso_smiles'])
compound_iso_smiles = set(compound_iso_smiles)
smile_graph = {}
for smile in compound_iso_smiles:
    g = smile_to_graph(smile)
    smile_graph[smile] = g

# TODO: continue

proteins = []
opts = ['train', 'test']
for opt in opts:
    df = pd.read_csv('data/davis_' + opt + '.csv')
    compound_iso_smiles += list(df['compound_iso_smiles'])
compound_iso_smiles = set(compound_iso_smiles)
smile_graph = {}
for smile in compound_iso_smiles:
    g = smile_to_graph(smile)
    smile_graph[smile] = g

processed_data_file_train = 'data/processed/davis_train.pt'
processed_data_file_test = 'data/processed/davis_test.pt'
if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test))):
    df = pd.read_csv('data/davis_train.csv')
    train_drugs, train_prots, train_Y = list(df['compound_iso_smiles']), list(df['target_sequence']), list(
        df['affinity'])
    XT = [seq_cat(t) for t in train_prots]
    train_drugs, train_prots, train_Y = np.asarray(train_drugs), np.asarray(XT), np.asarray(train_Y)
    df = pd.read_csv('data/davis_test.csv')
    test_drugs, test_prots, test_Y = list(df['compound_iso_smiles']), list(df['target_sequence']), list(
        df['affinity'])
    XT = [seq_cat(t) for t in test_prots]
    test_drugs, test_prots, test_Y = np.asarray(test_drugs), np.asarray(XT), np.asarray(test_Y)

    # make data PyTorch Geometric ready
    print('preparing davis_train.pt in pytorch format!')
    train_data = TestbedDataset(root='data', dataset='davis_train', xd=train_drugs, xt=train_prots, y=train_Y,
                                smile_graph=smile_graph)
    print('preparing davis_test.pt in pytorch format!')
    test_data = TestbedDataset(root='data', dataset='davis_test', xd=test_drugs, xt=test_prots, y=test_Y,
                               smile_graph=smile_graph)
    print(processed_data_file_train, ' and ', processed_data_file_test, ' have been created')
else:
    print(processed_data_file_train, ' and ', processed_data_file_test, ' are already created')