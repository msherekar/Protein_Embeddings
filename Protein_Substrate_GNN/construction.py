"""
Graph Construction Module
"""
import os
import rdkit
import torch
import pickle
import numpy as np
import networkx as nx
from rdkit import Chem
from rdkit.Chem import rdchem
from scipy.spatial import distance_matrix
from torch_geometric.data import Data

from preprocessing import MolecularPreprocessor

# Modular feature extraction (switch between basic and full atom features).
# Generalized interaction threshold (for different biomolecular systems).
# Supports different molecular representations (expandable to backbone/residue-level graphs).
class GraphConverter:
    def __init__(self, feature_set='basic', interaction_threshold=5.0):
        self.feature_set = feature_set
        self.interaction_threshold = interaction_threshold

    def atom_features(self, atom):
        """Extracts atomic features based on feature set."""
        if self.feature_set == 'basic':
            return [
                atom.GetAtomicNum(),
                atom.GetDegree(),
                atom.GetImplicitValence(),
                int(atom.GetIsAromatic())
            ]
        elif self.feature_set == 'full':
            return [
                atom.GetAtomicNum(),
                atom.GetDegree(),
                atom.GetImplicitValence(),
                int(atom.GetIsAromatic()),
                atom.GetFormalCharge(),
                int(atom.GetChiralTag())
            ]
        else:
            raise ValueError("Unsupported feature set")

    def mol_to_graph(self, mol):
        """Converts molecule into graph format."""
        graph = nx.Graph()
        for atom in mol.GetAtoms():
            graph.add_node(atom.GetIdx(), feats=torch.tensor(self.atom_features(atom), dtype=torch.float32))
        
        for bond in mol.GetBonds():
            graph.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())

        return graph

    def add_interactions(self, ligand, protein):
        """Adds interaction edges between ligand and protein atoms based on threshold."""
        pos_l = np.array(ligand.GetConformers()[0].GetPositions())
        pos_p = np.array(protein.GetConformers()[0].GetPositions())

        dis_matrix = distance_matrix(pos_l, pos_p)
        interaction_edges = np.where(dis_matrix < self.interaction_threshold)

        interaction_graph = nx.Graph()
        for i, j in zip(interaction_edges[0], interaction_edges[1]):
            interaction_graph.add_edge(i, j + ligand.GetNumAtoms())  # Shift protein indices

        return interaction_graph

if __name__ == '__main__':
    data_root = "./data/toy_set"
    preprocessor = MolecularPreprocessor(data_root, input_format="pdb")
    graph_converter = GraphConverter(feature_set='full', interaction_threshold=6.0)

    # Test loading
    ligand, protein = preprocessor.get_complex("6k4d")

    # Check results
    if ligand and protein:
        print("✅ Ligand and Protein loaded successfully!")
        ligand_graph = graph_converter.mol_to_graph(ligand)
        protein_graph = graph_converter.mol_to_graph(protein)
        interaction_graph = graph_converter.add_interactions(ligand, protein)
        print("✅ Graphs constructed successfully!")
    else:
        print("❌ Failed to load ligand or protein!")
