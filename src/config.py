import torch

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"

print(f"Using device: {DEVICE}")
