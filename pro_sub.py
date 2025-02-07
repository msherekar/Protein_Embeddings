
# Protein-Substrate Graph Embedding Code

import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
from rdkit import Chem
from Bio import PDB
import numpy as np
import networkx as nx

# Function to extract atomic features
def atom_features(atom):
    atom_type = ['C', 'N', 'O', 'S', 'P', 'H', 'F', 'Cl', 'Br', 'I']
    atom_one_hot = [int(atom.GetSymbol() == x) for x in atom_type]
    return torch.tensor(atom_one_hot, dtype=torch.float)

# Function to extract bond features
def bond_features(bond):
    bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                  Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
    bond_one_hot = [int(bond.GetBondType() == x) for x in bond_types]
    return torch.tensor(bond_one_hot, dtype=torch.float)

# Function to construct a molecular graph
def mol_to_graph(mol):
    atom_features_list = [atom_features(atom) for atom in mol.GetAtoms()]
    edge_index = []
    edge_features_list = []
    
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i])  # Bidirectional edges
        edge_features_list.append(bond_features(bond))
        edge_features_list.append(bond_features(bond))
    
    x = torch.stack(atom_features_list, dim=0)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.stack(edge_features_list, dim=0)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

# Function to extract features from a PDB file
def pdb_to_graph(pdb_file):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    
    atom_features_list = []
    edge_index = []
    
    atom_list = []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    atom_list.append(atom)
    
    num_atoms = len(atom_list)
    
    # Create adjacency based on spatial proximity (cutoff distance 5Å)
    for i in range(num_atoms):
        atom_i = atom_list[i]
        atom_features_list.append(torch.tensor([atom_i.element == 'C', atom_i.element == 'N',
                                                atom_i.element == 'O', atom_i.element == 'S'], dtype=torch.float))
        for j in range(i + 1, num_atoms):
            atom_j = atom_list[j]
            distance = atom_i - atom_j
            if distance < 5.0:  # Add an edge if within 5Å
                edge_index.append([i, j])
                edge_index.append([j, i])
    
    x = torch.stack(atom_features_list, dim=0)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    return Data(x=x, edge_index=edge_index)

# Define a Graph Neural Network (GNN)
class ProteinLigandGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(ProteinLigandGNN, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, torch.arange(x.size(0)))  # Global pooling
        return x

# Example usage: Load a PDB and Ligand (SMILES) and extract embeddings
pdb_file = "example_protein.pdb"  # Replace with your PDB file
ligand_smiles = "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"  # Replace with your ligand

# Convert PDB structure to graph
protein_graph = pdb_to_graph(pdb_file)

# Convert ligand to graph
mol = Chem.MolFromSmiles(ligand_smiles)
ligand_graph = mol_to_graph(mol)

# Concatenate protein and ligand graph features
protein_ligand_graph = Data(
    x=torch.cat([protein_graph.x, ligand_graph.x], dim=0),
    edge_index=torch.cat([protein_graph.edge_index, ligand_graph.edge_index + len(protein_graph.x)], dim=1)
)

# Initialize and run the GNN
gnn = ProteinLigandGNN(in_channels=4, hidden_channels=64, out_channels=32)
embedding = gnn(protein_ligand_graph)
print("Protein-Ligand Embedding:", embedding.shape)
