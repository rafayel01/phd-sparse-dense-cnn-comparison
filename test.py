import os, torch, nibabel as nib
from training_loop_sparse import dense_batch_to_sparse
import numpy as np
from dataset import BraTSDataset
from model import UNet
import MinkowskiEngine as ME
from torch.utils.data import Dataset, DataLoader
from mink_model import SparseUNet


def dice_score(pred, target, num_classes=4):
    smooth = 1e-5
    scores = []

    pred = (
        torch.tensor(pred, dtype=torch.int64, device=target.device)
        if isinstance(pred, np.ndarray)
        else pred
    )
    target = target.long()

    for cls in range(num_classes):
        pred_cls = (pred == cls).float()
        target_cls = (target == cls).float()

        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum()

        dice = (2.0 * intersection + smooth) / (union + smooth)
        scores.append(dice.item())

    return scores


def dense_batch_to_sparse(inputs, targets, device, thresh=0.0):
    """
    Convert a dense batch [B, C, D, H, W] to a batched SparseTensor
    and return flattened target labels aligned with outputs.F.
    """
    coords_list, feats_list, labels_list = [], [], []
    B = inputs.shape[0]
    for b in range(B):
        vol = inputs[b]  # [C, D, H, W]
        mask = vol.abs().sum(dim=0) > thresh  # foreground voxels
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


# ------------- paths --------------------------------------------------------
BEST_CKPT = "/home/rafayel/PhD/lightning/lightning_logs/version_7/checkpoints/epoch=0-step=42.ckpt"
TEST_LIST = [
    [
        "/home/rafayel/PhD/datasets/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_269/BraTS20_Training_269_t1.nii",
        "/home/rafayel/PhD/datasets/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_269/BraTS20_Training_269_t1ce.nii",
        "/home/rafayel/PhD/datasets/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_269/BraTS20_Training_269_t2.nii",
        "/home/rafayel/PhD/datasets/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_269/BraTS20_Training_269_flair.nii",
        "/home/rafayel/PhD/datasets/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_269/BraTS20_Training_269_seg.nii",
    ]
]  # list[list[str]] for test cases
OUT_DIR = "preds"
os.makedirs(OUT_DIR, exist_ok=True)


val_ds = BraTSDataset(TEST_LIST)

test_dl = DataLoader(val_ds, batch_size=1)

# ------------- data ---------------------------------------------------------
# ------------- model --------------------------------------------------------
model = SparseUNet(in_channels=4, out_channels=4)
model.load_state_dict(torch.load("/home/rafayel/PhD/checkpoints/model_weights_128.pth"))
model.eval().to("cuda")
criterion = torch.nn.CrossEntropyLoss()

for i, (dense_img, dense_gt) in enumerate(test_dl):
    case_id = f"case_{i:03d}"  # or parse from filename list
    dense_img = dense_img.cuda()  # [1,4,64,64,64]
    dense_gt = dense_gt.cuda()  # [1,64,64,64]
    print(f"dense_img: {dense_img.shape}")
    print(f"dense_gt: {dense_gt.shape}")
    # sparse conversion
    sparse_x, _ = dense_batch_to_sparse(dense_img, dense_gt, device="cuda")
    print("sparse_x shape:", sparse_x.F.shape)  # features: [N, C]
    print("sparse_x coords:", sparse_x.C.shape)
    # exit()
    with torch.no_grad():
        logits = model(sparse_x)
    # if isinstance(logits, tuple):
    #     logits = logits[0]
    print(f"logits shape: {logits.shape}")
    pred_dense = logits.dense()  # [1,C,64,64,64]
    print(pred_dense[0].shape)
    if isinstance(pred_dense, tuple):
        dense_tensor = pred_dense[0]  # first item is the volume
    else:
        dense_tensor = pred_dense
    # print(pred_dense)
    pred_mask = dense_tensor.argmax(1)  # [1,64,64,64]
    pred_mask = pred_mask.squeeze(0).cpu().numpy().astype(np.uint8)
    print(f"pred_mask shape: {pred_mask.shape}")
    # pred_mask = pred_dense.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)

    # Estimate loss and dice
    # logits_flat = dense_tensor.squeeze(0)
    # # [4, D, H, W] → [N, 4]
    # logits_flat = logits_flat.permute(1, 2, 3, 0).reshape(-1, 4)

    # # [1, D, H, W] → [N]
    # targets_flat = dense_gt.squeeze(0).reshape(-1).long()
    # print(f"{logits_flat.shape}")
    # print(f"{targets_flat.shape}")
    # loss = criterion(logits_flat, targets_flat)

    # dice = dice_score(pred_mask, dense_gt.squeeze(0), num_classes=4)
    # print(f"\n {case_id}  |  Loss: {loss.item():.4f}")
    # for c, d in enumerate(dice):
    #     print(f"  Dice class {c}: {d:.4f}")
    # ---- save prediction as NIfTI ------------------------------------------
    # use GT header/affine for perfect alignment
    gt_path = [p for p in TEST_LIST[i] if "_seg" in p.lower()][0]
    gt_nib = nib.load(gt_path)
    pred_nib = nib.Nifti1Image(pred_mask, affine=gt_nib.affine, header=gt_nib.header)

    nib.save(pred_nib, os.path.join(OUT_DIR, case_id + "_pred.nii.gz"))
    # optionally copy GT for side-by-side viewing
    nib.save(gt_nib, os.path.join(OUT_DIR, case_id + "_gt.nii.gz"))

    print(f"✔ saved {case_id}")


# ------------- run test -----------------------------------------------------
# trainer = Trainer(accelerator="auto", devices=1)
# results = trainer.test(model, datamodule=test_dm)
# print(results)
