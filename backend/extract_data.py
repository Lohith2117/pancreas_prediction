import os
import numpy as np
import nibabel as nib
import cv2
import json
from tqdm import tqdm

def extract_dataset(root_dir, output_dir):
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)
    
    with open(os.path.join(root_dir, "dataset.json"), 'r') as f:
        data_info = json.load(f)

    print("🚀 Pre-extracting 3D volumes into 2D NumPy slices...")
    
    for sample in tqdm(data_info["training"]):
        img_path = os.path.join(root_dir, sample["image"].replace("./", ""))
        lbl_path = os.path.join(root_dir, sample["label"].replace("./", ""))
        base_name = os.path.basename(img_path).replace(".nii.gz", "")

        img_data = nib.load(img_path).get_fdata()
        lbl_data = nib.load(lbl_path).get_fdata()

        # Find slices with actual pancreas/cancer
        z_indices = np.where(np.any(lbl_data > 0, axis=(0, 1)))[0]

        for z in z_indices:
            img_slice = img_data[:, :, z]
            lbl_slice = lbl_data[:, :, z]

            # Normalize and Resize ONCE here
            img_slice = np.clip(img_slice, -100, 200)
            img_slice = (img_slice - (-100)) / (200 - (-100))
            img_slice = cv2.resize(img_slice, (256, 256)).astype(np.float32)
            lbl_slice = cv2.resize(lbl_slice.astype(np.uint8), (256, 256), interpolation=cv2.INTER_NEAREST)

            # Save as fast binary files
            np.save(os.path.join(output_dir, "images", f"{base_name}_z{z}.npy"), img_slice)
            np.save(os.path.join(output_dir, "masks", f"{base_name}_z{z}.npy"), lbl_slice)

if __name__ == "__main__":
    extract_dataset(root_dir="./data", output_dir="./data_preextracted")