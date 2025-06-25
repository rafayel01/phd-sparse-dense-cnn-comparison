import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.model_selection import train_test_split

from dataset import BraTSDataset
from mink_model import SparseUNet
from training_loop import train_sparse

# Set seeds
np.random.seed(42)
torch.manual_seed(42)

# Constants
DATA_DIR = "/home/rafayel/PhD/datasets/BraTS2020_TrainingData"

# Load data paths
all_paths = []
for root, _, files in os.walk(DATA_DIR):
    if all(f.endswith(".nii") for f in files):
        all_paths.append([os.path.join(root, f) for f in files])

# Split into train and test
train_paths, val_paths = train_test_split(all_paths, test_size=0.1, random_state=42)

train_ds = BraTSDataset(train_paths)
val_ds = BraTSDataset(val_paths)

train_dl = DataLoader(train_ds, batch_size=4, shuffle=True, drop_last=True)
test_dl = DataLoader(val_ds, batch_size=4)

model = SparseUNet(in_channels=4, out_channels=4)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# Training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
train_sparse(model, train_dl, test_dl, optimizer, criterion, device, epochs=10)
