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
        # Sort to maintain consistent ordering of modalities
        paths = sorted(paths)
        modalities = []
        for path in paths[:-1]:
            img = nib.load(path).get_fdata()
            img = self._resize(img)
            modalities.append(img)
        image = np.stack(modalities, axis=0)
        mask = nib.load(paths[-1]).get_fdata()
        mask = self._resize(mask)
        mask = (mask > 0).astype(np.uint8)  # Binary mask
        return torch.tensor(image, dtype=torch.float32), torch.tensor(
            mask, dtype=torch.long
        )

    def _resize(self, volume):
        # Crop or pad to desired shape
        output = np.zeros(IMG_SIZE)
        shape = np.minimum(volume.shape, IMG_SIZE)
        slices = tuple(slice(0, s) for s in shape)
        output[slices] = volume[slices]
        return output
