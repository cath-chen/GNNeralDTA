import numpy as np
import networkx as nx
from Bio.PDB import MMCIFParser

def parse_cif_and_generate_graph(cif_file, threshold=5.0):
    """
    Parse a CIF file and generate a graph based on a distance threshold.

    Args:
        cif_file (str): Path to the CIF file.
        threshold (float): Distance threshold for edges.

    Returns:
        nx.Graph: Graph where nodes are residues, and edges are within the threshold.
    """
    parser = MMCIFParser()
    structure = parser.get_structure("protein", cif_file)

    # Extract residues and their alpha-carbon (CA) coordinates
    residues = []
    ca_coords = []

    for model in structure:
        for chain in model:
            for residue in chain:
                if 'CA' in residue:  # Use alpha-carbon to represent the residue
                    residues.append(residue)
                    ca_coords.append(residue['CA'].coord)

    # Calculate pairwise distances between alpha-carbons
    ca_coords = np.array(ca_coords)
    distances = np.linalg.norm(ca_coords[:, np.newaxis, :] - ca_coords[np.newaxis, :, :], axis=-1)

    # Create a graph with nodes as residues
    G = nx.Graph()

    for i, residue in enumerate(residues):
        res_id = (residue.parent.id, residue.id[1])  # Chain ID and residue ID
        G.add_node(i, res_id=res_id)

    # Add edges based on the distance threshold
    for i in range(len(residues)):
        for j in range(i + 1, len(residues)):
            if distances[i, j] < threshold:
                G.add_edge(i, j, weight=distances[i, j])

    return G

# Example usage
cif_file = "pdb_files/ACVR1.cif"  # Path to your CIF file
threshold = 5.0  # Distance threshold in Ångstroms
graph = parse_cif_and_generate_graph(cif_file, threshold)

# Visualize the graph (optional)
import matplotlib.pyplot as plt
nx.draw(graph, with_labels=True)
plt.show()

# Print basic graph info
print(f"Graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")