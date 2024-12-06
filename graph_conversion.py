import networkx as nx
import numpy as np
from Bio.PDB import MMCIFParser
from rdkit import Chem

resname_to_fasta = {"ALA": 'A', "CYS": 'C', "ASP": 'D', "GLU": 'E', "PHE": 'F', "GLY": 'G', "HIS": 'H', "ILE": 'I',
                    "LYS": 'K', "LEU": 'L', "MET": 'M', "ASN": 'N', "PYL": 'O', "PRO": 'P', "GLN": 'Q', "ARG": 'R',
                    "SER": 'S', "THR": 'T', "SEC": 'U', "VAL": 'V', "TRP": 'W', "TYR": 'Y'}


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


def cif_to_graph(cif_file, threshold=5.0):
    parser = MMCIFParser()
    structure = parser.get_structure("protein", cif_file)

    # Extract residues and their alpha-carbon (CA) coordinates
    residues = []
    ca_coords = []

    for model in structure:
        for chain in model:
            for residue in chain:
                if 'CA' in residue:  # Use alpha-carbon to represent the residue
                    residues.append(
                        np.array(one_of_k_encoding(resname_to_fasta[residue.resname], resname_to_fasta.values()),
                                 dtype=float))
                    ca_coords.append(residue['CA'].coord)

    # Calculate pairwise distances between alpha-carbons
    ca_coords = np.array(ca_coords)
    distances = np.linalg.norm(ca_coords[:, np.newaxis, :] - ca_coords[np.newaxis, :, :], axis=-1)

    # get adjacency matrix from distances without duplicate edges
    adjacency = distances < threshold
    adjacency = adjacency.astype(float)
    adjacency *= np.tri(*adjacency.shape, k=-1)

    # get edge_index and edge weights
    edge_index = np.stack(np.where(adjacency)).T
    weights = distances[edge_index[:, 0], edge_index[:, 1]]

    return len(residues), residues, edge_index, weights
