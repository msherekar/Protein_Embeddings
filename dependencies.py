import torch
import torch_geometric
import rdkit
import Bio

print("PyTorch version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
print("Torch-Geometric version:", torch_geometric.__version__)
print("RDKit version:", rdkit.__version__)