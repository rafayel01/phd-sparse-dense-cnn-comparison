import nibabel as nib
import torch
from torch.utils.data import Dataset
import numpy as np

IMG_SIZE = (128, 128, 128)


class BraTSDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = [file_path for file_path in file_paths if file_path]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        paths = self.file_paths[idx]

        # Find the segmentation label path (contains 'seg')
        seg_path = [p for p in paths if "seg" in p.lower()]
        if not seg_path:
            raise ValueError(f"No segmentation file found in: {paths}")
        seg_path = seg_path[0]  # Assuming only one label file per sample

        # Use all other files as modalities
        modal_paths = [p for p in paths if p != seg_path]
        # Optional: sort for consistent order
        modal_paths = sorted(modal_paths)

        # Load modalities
        modalities = []
        for path in modal_paths:
            img = nib.load(path).get_fdata()
            img = self._resize(img)
            modalities.append(img)
        image = np.stack(modalities, axis=0)  # [C, D, H, W]

        # Load segmentation mask
        mask = nib.load(seg_path).get_fdata()
        mask = self._resize(mask)
        mask = mask.astype(np.uint8)
        # mask = (mask > 0).astype(np.uint8)

        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.long)


    def _resize(self, volume):
        # Crop or pad to desired shape
        output = np.zeros(IMG_SIZE)
        shape = np.minimum(volume.shape, IMG_SIZE)
        slices = tuple(slice(0, s) for s in shape)
        output[slices] = volume[slices]
        return output
