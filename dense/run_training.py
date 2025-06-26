import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from dataset import BraTSDataset
from model import UNet
from training_loop import train


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        """
        logits: [B, C, D, H, W] - raw model output (not softmaxed)
        target: [B, D, H, W] - ground truth class indices (0...C-1)
        """
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)  # [B, C, D, H, W]
        target_onehot = F.one_hot(target, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()  # [B, C, D, H, W]

        dims = (0, 2, 3, 4)  # sum over batch and spatial dims
        intersection = torch.sum(probs * target_onehot, dims)
        denominator = torch.sum(probs + target_onehot, dims)

        dice = (2. * intersection + self.smooth) / (denominator + self.smooth)
        dice_loss = 1 - dice  # per-class loss
        return dice_loss.mean()
# Set seeds
np.random.seed(42)
torch.manual_seed(42)

# Constants
DATA_DIR = "/root/.cache/kagglehub/datasets/awsaf49/brats20-dataset-training-validation/versions/1/BraTS2020_TrainingData"

# Load data paths
all_paths = []
for root, _, files in os.walk(DATA_DIR):
    if all(f.endswith(".nii") for f in files):
        all_paths.append([os.path.join(root, f) for f in files])

# Split into train and test
train_paths, val_paths = train_test_split(all_paths, test_size=0.2, random_state=42)


train_ds = BraTSDataset(train_paths)
val_ds = BraTSDataset(val_paths)
train_dl = DataLoader(train_ds, batch_size=2, shuffle=True)
test_dl = DataLoader(val_ds, batch_size=2)


model = UNet(in_channels=4, out_channels=4)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = DiceLoss()

# Training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{device}")
train(model, train_dl, test_dl, optimizer, criterion, device, epochs=10)
