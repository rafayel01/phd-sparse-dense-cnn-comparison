import torch
import nibabel as nib
import numpy as np
import torch.nn.functional as F
from dense.model import UNet
from scipy.ndimage import zoom
import matplotlib.pyplot as plt

# ------------ CONFIG ------------------
CKPT_PATH = "/root/checkpoints/model_weights_dense.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 4
IMG_SIZE = (128, 128, 128)
# --------------------------------------


def resize_volume(volume, is_label=False):
    zoom_factors = [n / o for n, o in zip(IMG_SIZE, volume.shape)]
    order = 0 if is_label else 1  # Nearest for label, trilinear for image
    return zoom(volume, zoom=zoom_factors, order=order)


def load_nifti_modalities(modal_paths):
    vols = [resize_volume(nib.load(p).get_fdata()) for p in modal_paths]
    vols = np.stack(vols, axis=0)
    vols = torch.tensor(vols, dtype=torch.float32).unsqueeze(0)  # [1, 4, D, H, W]
    return vols


def run_dense_inference(dense_img, model, device):
    dense_img = dense_img.to(device)
    with torch.no_grad():
        logits = model(dense_img)  # [1, C, D, H, W]
        preds = logits.argmax(dim=1).squeeze(0).cpu().numpy()
        probs = F.softmax(logits, dim=1)
        print(f"{torch.max(probs, dim=1).values.squeeze().cpu().numpy() = }")
        # for i in range(NUM_CLASSES):
        #     plt.imshow(probs[0, i, :, :, 64].cpu(), cmap='hot')
        #     plt.title(f"Class {i} prob")
        #     plt.show()

    return logits, preds


def dice_per_class(logits, target, num_classes, smooth=1e-5):
    pred_probs = F.softmax(logits, dim=1)
    target_onehot = F.one_hot(target, num_classes).permute(0, 4, 1, 2, 3).float()
    dims = (0, 2, 3, 4)
    intersection = torch.sum(pred_probs * target_onehot, dims)
    union = torch.sum(pred_probs + target_onehot, dims)
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.cpu().numpy()


def save_nifti(volume_np, reference_path, save_path):
    ref_nib = nib.load(reference_path)
    out_nib = nib.Nifti1Image(
        volume_np.astype(np.uint8), affine=ref_nib.affine, header=ref_nib.header
    )
    nib.save(out_nib, save_path)


if __name__ == "__main__":
    test_modal_paths = [
        "/root/.cache/kagglehub/datasets/awsaf49/brats20-dataset-training-validation/versions/1/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_269/BraTS20_Training_269_t1.nii",
        "/root/.cache/kagglehub/datasets/awsaf49/brats20-dataset-training-validation/versions/1/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_269/BraTS20_Training_269_t1ce.nii",
        "/root/.cache/kagglehub/datasets/awsaf49/brats20-dataset-training-validation/versions/1/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_269/BraTS20_Training_269_t2.nii",
        "/root/.cache/kagglehub/datasets/awsaf49/brats20-dataset-training-validation/versions/1/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_269/BraTS20_Training_269_flair.nii",
    ]
    gt_path = "/root/.cache/kagglehub/datasets/awsaf49/brats20-dataset-training-validation/versions/1/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_269/BraTS20_Training_269_seg.nii"
    out_path = "/root/preds/prediction_dense_new_crop.nii"

    # Load model
    model = UNet(in_channels=4, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(CKPT_PATH))
    model.eval().to(DEVICE)

    # Load input image and GT
    dense_img = load_nifti_modalities(test_modal_paths)
    label_np = resize_volume(nib.load(gt_path).get_fdata(), is_label=True)
    label_tensor = torch.tensor(label_np, dtype=torch.long).unsqueeze(0).to(DEVICE)  # [1, D, H, W]

    # Inference
    logits, prediction = run_dense_inference(dense_img, model, DEVICE)
    print("Unique predicted labels:", np.unique(prediction))
    # Loss
    criterion = torch.nn.CrossEntropyLoss()
    ce_loss = criterion(logits, label_tensor)
    print(f"Cross-Entropy Loss: {ce_loss.item():.4f}")

    # Dice score
    dice_scores = dice_per_class(logits, label_tensor, num_classes=NUM_CLASSES)
    for i, d in enumerate(dice_scores):
        print(f"Dice (class {i}): {d:.4f}")

    # Save NIfTI prediction
    save_nifti(prediction, reference_path=gt_path, save_path=out_path)
    print("✔ Saved prediction:", out_path)
