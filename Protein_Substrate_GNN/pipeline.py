"""
End-to-End Embedding Pipeline
"""

import os
import pandas as pd
import torch
from torch_geometric.data import Data
from preprocessing import *
from construction import *

class EmbeddingPipeline:
    def __init__(self, data_dir, output_dir, feature_set='basic', interaction_threshold=5.0):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.preprocessor = MolecularPreprocessor(data_dir)
        self.graph_converter = GraphConverter(feature_set, interaction_threshold)

    def process_complex(self, complex_id, label):
        """Converts a ligand-protein complex into a graph and saves it."""
        ligand, protein = self.preprocessor.get_complex(complex_id)
        if ligand is None or protein is None:
            print(f"❌ Skipping {complex_id} due to missing molecule data.")
            return None

        ligand_graph = self.graph_converter.mol_to_graph(ligand)
        protein_graph = self.graph_converter.mol_to_graph(protein)
        interaction_graph = self.graph_converter.add_interactions(ligand, protein)

        if ligand_graph is None or protein_graph is None or interaction_graph is None:
            print(f"⚠️ Skipping complex {complex_id} due to graph conversion failure.")
            return None

        # Extract node features
        ligand_feats = torch.stack([feats for _, feats in ligand_graph.nodes(data="feats")])
        protein_feats = torch.stack([feats for _, feats in protein_graph.nodes(data="feats")])
        x = torch.cat([ligand_feats, protein_feats], dim=0)

        # Edge index construction
        num_ligand_atoms = ligand.GetNumAtoms()
        num_protein_atoms = protein.GetNumAtoms()

        ligand_edges = torch.stack([torch.tensor((u, v)) for u, v in ligand_graph.edges()], dim=1)
        protein_edges = torch.stack([torch.tensor((u + num_ligand_atoms, v + num_ligand_atoms)) for u, v in protein_graph.edges()], dim=1)
        edge_index_intra = torch.cat([ligand_edges, protein_edges], dim=-1)

        edge_index_inter = torch.stack([torch.tensor((u, v + num_ligand_atoms)) for u, v in interaction_graph.edges()], dim=1)

        # Save final graph data
        y = torch.tensor([label], dtype=torch.float32)
        data = Data(x=x, edge_index_intra=edge_index_intra, edge_index_inter=edge_index_inter, y=y)
        torch.save(data, os.path.join(self.output_dir, f"{complex_id}.pt"))

    def process_all(self, dataset_df):
        """Processes all complexes in a dataset."""
        for _, row in dataset_df.iterrows():
            self.process_complex(row['pdbid'], row['-logKd/Ki'])

if __name__ == '__main__':
    data_root = "./data/toy_set"
    output_dir = "./processed_graphs"
    os.makedirs(output_dir, exist_ok=True)

    dataset_df = pd.read_csv(os.path.join(data_root, "toy_examples.csv"))
    pipeline = EmbeddingPipeline(data_root, output_dir, feature_set='full', interaction_threshold=6.0)

    pipeline.process_all(dataset_df)
