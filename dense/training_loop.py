import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import csv
import os


def log_training_results_csv(csv_path, epoch, train_loss, val_loss, train_dice, val_dice):
    fieldnames = ['epoch', 'train_loss', 'val_loss'] + \
                 [f'train_dice_class_{i}' for i in range(len(train_dice))] + \
                 [f'val_dice_class_{i}' for i in range(len(val_dice))]

    # Create the file with headers if it doesn't exist
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode='a', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        row = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
        }
        for i, d in enumerate(train_dice):
            row[f'train_dice_class_{i}'] = d
        for i, d in enumerate(val_dice):
            row[f'val_dice_class_{i}'] = d
        writer.writerow(row)


def dice_score(pred, target, num_classes):
    dice = []
    pred = pred.detach().cpu()
    target = target.detach().cpu()
    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds & target_inds).sum().item()
        union = pred_inds.sum().item() + target_inds.sum().item()
        if union == 0:
            dice.append(1.0)
        else:
            dice.append(2.0 * intersection / union)
    return dice


def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    epochs=20,
    num_classes=4,
):
    model.to(device)
    min_loss = np.inf
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_dice = np.zeros(num_classes)

        print(f"\nEpoch {epoch+1}/{epochs}")
        for inputs, targets in tqdm(train_loader, desc="Training"):
            inputs = inputs.to(device, dtype=torch.float32)  # [B, 4, D, H, W]
            targets = targets.to(device, dtype=torch.long)  # [B, D, H, W]
            # print(f"{targets.max() = }")
            # exit()

            targets[targets == 4] = 3
            optimizer.zero_grad()
            outputs = model(inputs)  # [B, C, D, H, W]

            loss = criterion(outputs, targets)
            # print("Loss value:", loss)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            preds = outputs.argmax(dim=1)  # [B, D, H, W]
            batch_dice = dice_score(preds, targets, num_classes)
            train_dice += np.array(batch_dice)

        avg_train_loss = train_loss / len(train_loader)
        avg_train_dice = train_dice / len(train_loader)

        print(f"Train Loss: {avg_train_loss:.4f}")
        for cls in range(num_classes):
            print(f"  Dice (class {cls}): {avg_train_dice[cls]:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        val_dice = np.zeros(num_classes)

        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc="Validating"):
                inputs = inputs.to(device, dtype=torch.float32)
                targets = targets.to(device, dtype=torch.long)
                targets[targets == 4] = 3
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

                preds = outputs.argmax(dim=1)
                batch_dice = dice_score(preds, targets, num_classes)
                val_dice += np.array(batch_dice)

        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice / len(val_loader)

        print(f"Val Loss: {avg_val_loss:.4f}")
        for cls in range(num_classes):
            print(f"  Val Dice (class {cls}): {avg_val_dice[cls]:.4f}")
        
        log_training_results_csv(
            csv_path="/root/logs/training_metrics_dense.csv",  # adjust path as needed
            epoch=epoch + 1,
            train_loss=avg_train_loss,
            val_loss=avg_val_loss,
            train_dice=avg_train_dice.tolist(),
            val_dice=avg_val_dice.tolist()
        )
        if avg_val_loss < min_loss:
            print("Saved new checkpoint!")
            min_loss = avg_val_loss
            torch.save(
                model.state_dict(),
                "/root/checkpoints/model_weights_dense.pth",
            )
