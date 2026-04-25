import os
import torch
import numpy as np
from torch.utils.data import Dataset

class PancreaticCTDataset(Dataset):
    def __init__(self, root_dir, split="train"):
        # This will now look inside data_preextracted/images
        self.image_dir = os.path.join(root_dir, "images")
        self.mask_dir = os.path.join(root_dir, "masks")
        # Get all the .npy files we just created
        self.filenames = [f for f in os.listdir(self.image_dir) if f.endswith(".npy")]
        print(f"✅ Dataset loaded: {len(self.filenames)} slices found.")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        
        # Super fast loading from SSD
        img_slice = np.load(os.path.join(self.image_dir, fname))
        lbl_slice = np.load(os.path.join(self.mask_dir, fname))

        # Convert to Tensors (3-channel for ResNet backbone)
        img_tensor = torch.from_numpy(img_slice).unsqueeze(0).repeat(3, 1, 1)
        mask_tensor = torch.from_numpy(lbl_slice).float().unsqueeze(0)
        
        # Classification label (1 if tumor/pancreas is present)
        label = 1 if np.any(lbl_slice > 0) else 0

        return {"image": img_tensor, "mask": mask_tensor, "label": torch.tensor(label)}