import torch
import nibabel as nib
import numpy as np
from dense.model import UNet  # your dense UNet model


# ------------ CONFIG ------------------
CKPT_PATH = "/root/checkpoints/model_weights_dense.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 4
# --------------------------------------


def load_nifti_modalities(modal_paths):
    """Load 4 modalities and return [1, 4, D, H, W] tensor"""
    vols = [nib.load(p).get_fdata() for p in modal_paths]
    vols = np.stack(vols, axis=0)  # [4, D, H, W]
    vols = torch.tensor(vols, dtype=torch.float32).unsqueeze(0)  # [1, 4, D, H, W]
    return vols


def run_dense_inference(dense_img, model, device):
    """Run inference using standard dense UNet"""
    dense_img = dense_img.to(device)
    with torch.no_grad():
        logits = model(dense_img)  # [1, C, D, H, W]
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()  # [D, H, W]
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
        "/root/.cache/kagglehub/datasets/awsaf49/brats20-dataset-training-validation/versions/1/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_269/BraTS20_Training_269_t1.nii",
        "/root/.cache/kagglehub/datasets/awsaf49/brats20-dataset-training-validation/versions/1/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_269/BraTS20_Training_269_t1ce.nii",
        "/root/.cache/kagglehub/datasets/awsaf49/brats20-dataset-training-validation/versions/1/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_269/BraTS20_Training_269_t2.nii",
        "/root/.cache/kagglehub/datasets/awsaf49/brats20-dataset-training-validation/versions/1/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_269/BraTS20_Training_269_flair.nii",
        # "/root/.cache/kagglehub/datasets/awsaf49/brats20-dataset-training-validation/versions/1/BraTS2020_TrainingData/BraTS20_Training_269/BraTS20_Training_269_seg.nii",
    ]
    gt_path = "/root/.cache/kagglehub/datasets/awsaf49/brats20-dataset-training-validation/versions/1/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_269/BraTS20_Training_269_seg.nii"
    out_path = "/root/preds/prediction_dense.nii"

    # Load model
    model = UNet(in_channels=4, out_channels=NUM_CLASSES)
    model.load_state_dict(torch.load(CKPT_PATH))
    model.eval().to(DEVICE)

    # Load input
    dense_img = load_nifti_modalities(test_modal_paths)

    # Predict
    prediction = run_dense_inference(dense_img, model, DEVICE)

    # Save
    save_nifti(prediction, reference_path=gt_path, save_path=out_path)
    print("✔ Dense UNet inference complete:", out_path)
