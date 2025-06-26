import torch
import MinkowskiEngine as ME
from mink_model import SparseUNet  # your custom model
import nibabel as nib
import numpy as np


# ------------ CONFIG ------------------
CKPT_PATH = "path/to/your_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = (128, 128, 128)
NUM_CLASSES = 4
# --------------------------------------


def dense_batch_to_sparse(inputs, targets=None, device="cuda", thresh=0.0):
    """
    Convert a dense batch [B, C, D, H, W] to a MinkowskiEngine SparseTensor.
    If `targets` is given, returns the corresponding flattened labels.

    Parameters:
        inputs : torch.Tensor [B, C, D, H, W]
        targets: torch.Tensor [B, D, H, W] or None
        device : str or torch.device
        thresh : float, optional threshold to filter out zero-background voxels

    Returns:
        sparse_tensor : ME.SparseTensor
        labels        : torch.Tensor [N] or None
    """
    coords_list, feats_list, labels_list = [], [], []

    B = inputs.shape[0]

    for b in range(B):
        vol = inputs[b]  # [C, D, H, W]
        mask = vol.abs().sum(dim=0) > thresh  # [D, H, W] foreground mask
        idx = mask.nonzero(as_tuple=False)  # [N, 3] (z, y, x)

        if idx.numel() == 0:
            # Avoid empty input — put dummy voxel
            idx = torch.zeros((1, 3), dtype=torch.long, device=device)
            feats = torch.zeros((1, vol.shape[0]), device=device)
            lbls = (
                torch.zeros(1, dtype=torch.long, device=device)
                if targets is not None
                else None
            )
        else:
            feats = vol[:, idx[:, 0], idx[:, 1], idx[:, 2]].T  # [N, C]
            lbls = (
                targets[b, idx[:, 0], idx[:, 1], idx[:, 2]]
                if targets is not None
                else None
            )

        batch_coords = torch.cat(
            [
                torch.full((idx.size(0), 1), b, dtype=torch.int32, device=device),
                idx.int(),
            ],
            dim=1,
        )  # [N, 1+3]

        coords_list.append(batch_coords)
        feats_list.append(feats)
        if lbls is not None:
            labels_list.append(lbls)

    coords = torch.cat(coords_list)  # [∑N, 4]
    feats = torch.cat(feats_list)  # [∑N, C]
    labels = torch.cat(labels_list) if labels_list else None

    sparse_tensor = ME.SparseTensor(
        features=feats.to(device), coordinates=coords.to(device)
    )
    return sparse_tensor, labels.to(device) if labels is not None else None


def load_nifti_modalities(modal_paths):
    """Load 4 modalities and return [1, 4, D, H, W] tensor"""
    vols = [nib.load(p).get_fdata() for p in modal_paths]
    vols = np.stack(vols, axis=0)  # [4, D, H, W]
    vols = torch.tensor(vols, dtype=torch.float32).unsqueeze(0)  # [1,4,D,H,W]
    return vols


def run_inference(dense_img, model, device):
    """Run inference given dense input [1, C, D, H, W]"""
    dense_img = dense_img.to(device)

    # Dummy ground truth for sparse conversion
    dummy_gt = torch.zeros(
        dense_img.shape[0], *dense_img.shape[2:], dtype=torch.long
    ).to(device)

    # Convert to sparse
    sparse_x, _ = dense_batch_to_sparse(dense_img, dummy_gt, device)

    with torch.no_grad():
        logits = model(sparse_x)
        dense_logits = logits.dense(sparse_shape=dense_img.shape[2:])  # [1, C, D, H, W]
        pred = dense_logits.argmax(dim=1).squeeze(0).cpu().numpy()  # [D, H, W]
    return pred


def save_nifti(volume_np, reference_path, save_path):
    """Save prediction as NIfTI using reference header/affine"""
    ref_nib = nib.load(reference_path)
    out_nib = nib.Nifti1Image(
        volume_np.astype(np.uint8), affine=ref_nib.affine, header=ref_nib.header
    )
    nib.save(out_nib, save_path)


if __name__ == "__main__":
    # Example BraTS test case
    test_modal_paths = [
        "BraTS..._t1.nii",
        "BraTS..._t1ce.nii",
        "BraTS..._t2.nii",
        "BraTS..._flair.nii",
    ]
    gt_path = "BraTS..._seg.nii"
    out_path = "prediction.nii.gz"

    # Load model
    model = SparseUNet(in_channels=4, out_channels=NUM_CLASSES)
    model.load_state_dict(torch.load(CKPT_PATH))
    model.eval().to(DEVICE)

    # Load input image
    dense_img = load_nifti_modalities(test_modal_paths)

    # Run prediction
    prediction = run_inference(dense_img, model, DEVICE)

    # Save result
    save_nifti(prediction, reference_path=gt_path, save_path=out_path)
    print("✔ Inference complete:", out_path)
