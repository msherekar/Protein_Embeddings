"""
Molecular preprocessing module
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

class MolecularPreprocessor:
    def __init__(self, data_dir, input_format='pdb'):
        self.data_dir = data_dir
        self.input_format = input_format

    def load_molecule(self, file_path):
        """Loads a molecule and handles errors gracefully."""
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return None
        
        mol = None
        if file_path.endswith(".pdb"):
            mol = Chem.MolFromPDBFile(file_path, removeHs=True)
        elif file_path.endswith(".mol2"):
            mol = Chem.MolFromMol2File(file_path, removeHs=True)

        if mol is None:
            print(f"⚠️ RDKit failed to read: {file_path}")
        return mol

    def get_complex(self, complex_id):
        """Loads ligand and protein for a given complex ID."""
        complex_dir = os.path.join(self.data_dir, complex_id)
        ligand_path = os.path.join(complex_dir, f"{complex_id}_ligand.{self.input_format}")
        protein_path = os.path.join(complex_dir, f"{complex_id}_protein.{self.input_format}")  

        ligand = self.load_molecule(ligand_path)
        protein = self.load_molecule(protein_path)

        if ligand is None or protein is None:
            print(f"❌ Error loading complex {complex_id}")
            return None, None
        return ligand, protein

if __name__ == '__main__':
    data_root = "./data/toy_set"
    preprocessor = MolecularPreprocessor(data_root, input_format="pdb")

    # Test loading
    ligand, protein = preprocessor.get_complex("6k4d")

    # Check results
    if ligand:
        print("✅ Ligand loaded successfully!")
    else:
        print("❌ Failed to load ligand!")

    if protein:
        print("✅ Protein loaded successfully!")
    else:
        print("❌ Failed to load protein!")
