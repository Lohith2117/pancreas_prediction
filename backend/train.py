import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from src.model import PancreaticDetectionNet, CombinedLoss
from src.preprocessing import PancreaticCTDataset 
from tqdm import tqdm  
import os

# Essential for Mac M2 to handle specific GPU operations
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

def train_model(data_path, epochs=10):
    if not os.path.exists("models"):
        os.makedirs("models")

    # Detection for Apple Silicon GPU
    device = torch.device(
        "mps" if torch.backends.mps.is_available() 
        else "cuda" if torch.cuda.is_available() 
        else "cpu"
    )
    
    model = PancreaticDetectionNet(pretrained=True).to(device)
    criterion = CombinedLoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)

    # 2. Optimized Data Loader for M2 Air
    train_ds = PancreaticCTDataset("./data_preextracted", split="train")    
    # FIX: num_workers=0 and pin_memory=False prevents memory "bottlenecking" on Mac
    # In backend/train.py
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)

    print(f"🚀 Starting training on: {str(device).upper()}")
    print(f"📦 Total slices: {len(train_ds)}")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in progress_bar:
            # Move data to GPU (MPS)
            imgs = batch["image"].to(device)
            labels = batch["label"].to(device)
            masks = batch["mask"].to(device)

            # Training steps
            cls_logit, seg_pred = model(imgs)
            loss = criterion(cls_logit, seg_pred, labels, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)
        print(f"✅ Epoch {epoch+1} Complete - Avg Loss: {avg_loss:.4f}")
        torch.save({'model_state': model.state_dict()}, "models/best_model.pt")
    
    print("✨ Success! 'models/best_model.pt' has been created.")

if __name__ == "__main__":
    train_model(data_path="./data.preextracted")