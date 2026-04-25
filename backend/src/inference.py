import torch
import cv2
import numpy as np
import nibabel as nib
import io
import gzip
import pydicom  # <-- New Import: run 'pip install pydicom'
from src.model import PancreaticDetectionNet

class PancreasInference:
    def __init__(self, model_path):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = PancreaticDetectionNet(pretrained=False).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.eval()

    def predict(self, image_bytes, filename="file.jpg"):
        import nibabel as nib
        import pydicom
        import io
        import gzip

        try:
            filename_lower = filename.lower()

            # 1. HANDLE DICOM (.dcm)
            if filename_lower.endswith('.dcm'):
                ds = pydicom.dcmread(io.BytesIO(image_bytes))
                img = ds.pixel_array.astype(np.float32)
                intercept = ds.RescaleIntercept if 'RescaleIntercept' in ds else -1024
                slope = ds.RescaleSlope if 'RescaleSlope' in ds else 1
                img = img * slope + intercept
                # Windowing
                img = np.clip(img, -100, 200)
                img = ((img - (-100)) / 300.0 * 255).astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            # 2. HANDLE NIFTI (.nii, .nii.gz)
            elif filename_lower.endswith(('.nii', '.gz')):
                if filename_lower.endswith('.gz'):
                    with gzip.GzipFile(fileobj=io.BytesIO(image_bytes)) as gz:
                        decompressed_data = gz.read()
                else:
                    decompressed_data = image_bytes
                
                img_nifti = nib.Nifti1Image.from_bytes(decompressed_data)
                vol = img_nifti.get_fdata()
                img = vol[..., vol.shape[-1] // 2]
                img = np.clip(img, -100, 200)
                img = ((img - (-100)) / 300.0 * 255).astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            # 3. HANDLE STANDARD IMAGES (JPG, PNG, JPEG)
            else:
                nparr = np.frombuffer(image_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                # Note: JPGs from case reports are already "visualized"
                # so we don't apply medical windowing to them.

            if img is None:
                raise ValueError(f"Could not decode image: {filename}")

            # --- PREDICTION ---
            img_resized = cv2.resize(img, (256, 256))
            x = img_resized.astype(np.float32) / 255.0
            x = torch.from_numpy(x).permute(2, 0, 1).float().unsqueeze(0).to(self.device)

            with torch.no_grad():
                logit, mask = self.model(x)
                prob = torch.sigmoid(logit).item()
                mask_np = torch.sigmoid(mask).squeeze().cpu().numpy()
            
            return prob, mask_np, img

        except Exception as e:
            print(f"❌ Inference Error: {e}")
            raise e