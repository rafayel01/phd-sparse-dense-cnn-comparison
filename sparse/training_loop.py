import MinkowskiEngine as ME
import torch
import numpy as np
from tqdm import tqdm


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


# ---------- helper -----------------------------------------------------------
def dense_batch_to_sparse(inputs, targets, device, thresh=0.0):
    """
    Convert a dense batch [B, C, D, H, W] to a batched SparseTensor
    and return flattened target labels aligned with outputs.F.
    """
    coords_list, feats_list, labels_list = [], [], []
    B = inputs.shape[0]
    for b in range(B):
        vol = inputs[b]  # [C, D, H, W]
        mask = vol.abs().sum(dim=1) > thresh  # foreground voxels
        idx = mask.nonzero(as_tuple=False)  # [N, 3] (z, y, x)
        # idx = torch.stack(
        #     torch.meshgrid(
        #         torch.arange(vol.shape[1], device=vol.device),
        #         torch.arange(vol.shape[2], device=vol.device),
        #         torch.arange(vol.shape[3], device=vol.device),
        #     ),
        #     dim=-1,
        # ).reshape(
        #     -1, 3
        # )  # [N, 3]
        if idx.numel() == 0:  # fully empty — guard
            # put a dummy voxel at (0,0,0) so MinkowskiEngine won't crash
            idx = torch.zeros((1, 3), dtype=torch.long, device=inputs.device)
            feats = torch.zeros((1, inputs.size(1)), device=inputs.device)
            lbls = torch.zeros(1, dtype=torch.long, device=inputs.device)
        else:
            feats = vol[:, idx[:, 0], idx[:, 1], idx[:, 2]].T  # [N, C]
            lbls = targets[b, idx[:, 0], idx[:, 1], idx[:, 2]]

        # prepend batch index
        batch_coords = torch.cat(
            [
                torch.full(
                    (idx.size(0), 1), b, dtype=torch.int32, device=inputs.device
                ),
                idx.int(),
            ],
            dim=1,
        )  # [N, 1+3]

        coords_list.append(batch_coords)
        feats_list.append(feats)
        labels_list.append(lbls)

    coords = torch.cat(coords_list)  # [∑N, 4]
    feats = torch.cat(feats_list)  # [∑N, C]
    labels = torch.cat(labels_list)  # [∑N]

    sparse_in = ME.SparseTensor(
        features=feats.float().to(device), coordinates=coords.to(device)
    )
    return sparse_in, labels.to(device)


# ---------- sparse train loop ------------------------------------------------
def train_sparse(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    epochs=20,
    num_classes=4,
    thresh=0.0,  # intensity threshold for sparsification
):
    model.to(device)
    min_loss = np.inf
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        # ------------------- training ---------------------------------------
        model.train()
        train_loss, train_dice = 0.0, np.zeros(num_classes)

        for dense_x, dense_y in tqdm(train_loader, desc="Training"):
            dense_x = dense_x.to(device, dtype=torch.float32)  # [B, 4, D, H, W]
            dense_y = dense_y.to(device, dtype=torch.long)  # [B, D, H, W]
            sparse_x, flat_y = dense_batch_to_sparse(dense_x, dense_y, device, thresh)
            # print("sparse_x shape:", sparse_x.F.shape)  # features: [N, C]
            # print("sparse_x coords:", sparse_x.C.shape)
            optimizer.zero_grad()
            out = model(sparse_x)  # SparseTensor
            loss = criterion(out.F, flat_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            preds = out.F.argmax(dim=1)  # [N_active]
            batch_dice = dice_score(preds, flat_y, num_classes)  # your util
            train_dice += np.array(batch_dice)

        avg_train_loss = train_loss / len(train_loader)
        avg_train_dice = train_dice / len(train_loader)

        print(f"Train Loss: {avg_train_loss:.4f}")
        for cls in range(num_classes):
            print(f"  Dice (class {cls}): {avg_train_dice[cls]:.4f}")

        # ------------------- validation ------------------------------------
        model.eval()
        val_loss, val_dice = 0.0, np.zeros(num_classes)
        with torch.no_grad():
            for dense_x, dense_y in tqdm(val_loader, desc="Validating"):
                dense_x = dense_x.to(device, dtype=torch.float32)
                dense_y = dense_y.to(device, dtype=torch.long)

                sparse_x, flat_y = dense_batch_to_sparse(
                    dense_x, dense_y, device, thresh
                )

                out = model(sparse_x)
                loss = criterion(out.F, flat_y)
                val_loss += loss.item()

                preds = out.F.argmax(dim=1)
                batch_dice = dice_score(preds, flat_y, num_classes)
                val_dice += np.array(batch_dice)
        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice / len(val_loader)
        if avg_val_loss < min_loss:
            print("Saved new checkpoint!")
            min_loss = avg_val_loss
            torch.save(
                model.state_dict(),
                "/home/rafayel/PhD/checkpoints/model_weights_128.pth",
            )
        print(f"Val  Loss: {avg_val_loss:.4f}")
        print(f"Val  Dice: {avg_val_dice}")
        for cls in range(num_classes):
            print(f"  Val Dice (class {cls}): {avg_val_dice[cls]:.4f}")
