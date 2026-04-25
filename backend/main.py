import cv2
import numpy as np
import base64
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from src.inference import PancreasInference
import uvicorn

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

predictor = PancreasInference(model_path="models/best_model.pt")

@app.post("/predict")
async def predict_pancreas(file: UploadFile = File(...)):
    image_bytes = await file.read()
    probability, mask_np, original_slice = predictor.predict(image_bytes, filename=file.filename)
    
    # Encode Original Slice
    _, buffer = cv2.imencode('.jpg', original_slice)
    img_b64 = base64.b64encode(buffer).decode('utf-8')

    # Encode Mask (Red Overlay)
    mask_visual = np.zeros((*mask_np.shape, 4), dtype=np.uint8)
    # Change this line in main.py
    mask_visual[mask_np > 0.5] = [0, 0, 255, 160] # [B, G, R, Alpha] - OpenCV uses BGR!
    _, m_buffer = cv2.imencode('.png', mask_visual)
    mask_b64 = base64.b64encode(m_buffer).decode('utf-8')
    
    return {
        "probability": float(probability),
        "image": f"data:image/jpeg;base64,{img_b64}",
        "mask": f"data:image/png;base64,{mask_b64}"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)