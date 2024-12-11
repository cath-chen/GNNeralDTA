import networkx as nx
import numpy as np
from Bio.PDB import FastMMCIFParser
from rdkit import Chem
import json

resname_to_fasta = {"ALA": 'A', "CYS": 'C', "ASP": 'D', "GLU": 'E', "PHE": 'F', "GLY": 'G', "HIS": 'H', "ILE": 'I',
                    "LYS": 'K', "LEU": 'L', "MET": 'M', "ASN": 'N', "PYL": 'O', "PRO": 'P', "GLN": 'Q', "ARG": 'R',
                    "SER": 'S', "THR": 'T', "SEC": 'U', "VAL": 'V', "TRP": 'W', "TYR": 'Y', "any": 'X'}


def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)

    c_size = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])

    return c_size, features, edge_index


def cif_to_graph(cif_file, sequence = None, threshold=8.0):
    parser = FastMMCIFParser()
    structure = parser.get_structure("protein", cif_file)

    # Extract residues and their alpha-carbon (CA) coordinates
    residues = []
    coords = []

    for model in structure:
        for chain in model:
            for residue in chain:
                residues.append(resname_to_fasta.get(residue.resname, 'X'))
                if 'CA' in residue:  # Use alpha-carbon to represent the residue
                    coords.append(residue['CA'].coord)
                else:
                    for atom in residue:
                        coords.append(atom.coord)
                        break

    skip_coords = False
    residues = ''.join(residues)
    if sequence is not None:
        if sequence in residues:
            start = residues.index(sequence)
            end = start + len(sequence)
            coords = coords[start:end]
        else:
            skip_coords = True
        residues = sequence

    residues = [np.array(one_of_k_encoding(residue, resname_to_fasta.values()), dtype=float) for residue in residues]

    # Calculate pairwise distances between alpha-carbons
    if not skip_coords:
        coords = np.array(coords)
        distances = np.linalg.norm(coords[:, np.newaxis, :] - coords[np.newaxis, :, :], axis=-1)
    else:
        distances = np.ones((len(residues), len(residues))) * threshold + 1

    # get adjacency matrix from distances without duplicate edges
    adjacency = distances < threshold
    adjacency = adjacency.astype(float)
    adjacency *= np.tri(*adjacency.shape, k=-1)  # remove double edges
    np.fill_diagonal(adjacency[1:, :], 1)  # make sure neighbors are actually connected

    # get edge_index and edge weights
    edge_index = np.stack(np.where(adjacency)).T
    if not skip_coords:
        weights = distances[edge_index[:, 0], edge_index[:, 1]]
    else:
        weights = np.ones(len(residues) - 1) * 3.8

    assert all(np.diag(adjacency, k=-1))

    if not skip_coords:
        print(f"converted {cif_file}")
    else:
        print(f"converted {cif_file} without coords")

    return len(residues), residues, edge_index, weights
