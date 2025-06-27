import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os

# Path to one subject directory
sample_subject_dir = "/root/.cache/kagglehub/datasets/awsaf49/brats20-dataset-training-validation/versions/1/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_269/"

# Define file names
modalities = {
    "T1": "BraTS20_Training_269_t1.nii",
    "T1CE": "BraTS20_Training_269_t1ce.nii",
    "T2": "BraTS20_Training_269_t2.nii",
    "FLAIR": "BraTS20_Training_269_flair.nii",
    "Groud_truth": "BraTS20_Training_269_seg.nii",
}
path_to_pred = "/root/preds"
label_file = "prediction_dense_new_crop.nii"

# Create output folder
output_dir = "output_modalities_combined"
os.makedirs(output_dir, exist_ok=True)

# Load and normalize modalities
slices = []
for name, filename in modalities.items():
    img_path = os.path.join(sample_subject_dir, filename)
    img = nib.load(img_path).get_fdata()

    mid_slice = img.shape[2] // 2
    print(f"Image shape: {img.shape}, Mid slice: {mid_slice}")

    slice_img = img[:, :, mid_slice]
    slice_img = (slice_img - np.min(slice_img)) / (
        np.max(slice_img) - np.min(slice_img) + 1e-8
    )
    slices.append((name, slice_img))

# Load segmentation mask
label_path = os.path.join(path_to_pred, label_file)
label_img = nib.load(label_path).get_fdata()
mid_slice = label_img.shape[2] // 2

print(f"Image shape: {label_img.shape}, Mid slice: {mid_slice}")

label_slice = label_img[:, :, mid_slice]

# Optionally normalize or convert to integer
label_slice = label_slice.astype(np.uint8)

# Append label
slices.append(("Segmentation", label_slice))

# Plot all
fig, axes = plt.subplots(1, len(slices), figsize=(20, 4))
for ax, (name, img) in zip(axes, slices):
    if name == "Segmentation":
        ax.imshow(img, cmap="nipy_spectral", interpolation="none")  # distinct colormap
    else:
        ax.imshow(img, cmap="gray")
    ax.set_title(name)
    ax.axis("off")

# Save figure
combined_path = os.path.join(output_dir, "compare_gr_model_new_crop.png")
plt.tight_layout()
plt.savefig(combined_path)
plt.close()

print(f"Saved image with modalities + segmentation to {combined_path}")
