
# PancreasNet: AI-Powered PDAC Detection

PancreasNet is a professional-grade medical imaging workstation prototype designed to detect Pancreatic Ductal Adenocarcinoma (PDAC) from CT scans. It leverages a ResNet-50 backbone trained on the Medical Segmentation Decathlon dataset to provide real-time probability scores and visual segmentation masks.



## 🚀 Features

* **Universal File Support**: Seamlessly handles NIfTI (`.nii`, `.nii.gz`), DICOM (`.dcm`), and standard images (`.jpg`, `.png`).
* **Dual-Viewer Interface**: Side-by-side comparison of raw source scans and AI-generated segmentation overlays.
* **Hounsfield Unit Normalization**: Automated windowing (-100 to 200 HU) specifically calibrated for pancreatic tissue density.
* **High Confidence Inference**: Optimized for deployment on Apple Silicon (M-series) using Metal Performance Shaders (MPS).

## 🛠️ Tech Stack

* **Frontend**: React (Vite), Axios, Inline CSS for high-performance medical UI.
* **Backend**: FastAPI, Uvicorn, Python 3.x.
* **AI/ML**: PyTorch, Torchvision, OpenCV, Nibabel (NIfTI), PyDicom (DICOM).

## 📂 Project Structure

```text
pancreas-net/
├── backend/
│   ├── src/
│   │   ├── model.py          # ResNet-50 Neural Network architecture
│   │   └── inference.py      # Multi-format prediction logic
│   ├── models/
│   │   └── best_model.pt     # Trained weights (Note: Ignored by Git)
│   ├── data/                 # Raw clinical data storage
│   └── main.py               # FastAPI entry point
├── frontend/
│   ├── src/
│   │   ├── App.jsx           # Diagnostic Dashboard UI
│   │   └── main.jsx
│   └── package.json
└── requirements.txt          # Python dependencies
```

## ⚙️ Installation

### 1. Backend Setup
```bash
cd backend
pip install -r ../requirements.txt
# Ensure you have your best_model.pt in the backend/models/ folder
python main.py
```

### 2. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

## 🔬 Clinical Workflow
1.  **Select Source**: Upload a raw CT volume or a single slice.
2.  **Execute Analysis**: The backend normalizes the data and runs it through the `PancreaticDetectionNet`.
3.  **Review Findings**: Observe the probability index and the red segmentation mask highlighting suspicious lesions.

## 📜 Disclaimer
This project is for educational and research purposes only. It is not intended for clinical use or professional medical diagnosis.
