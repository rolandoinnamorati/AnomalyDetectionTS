import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.autoencoder import TimeSeriesAutoencoder
from utils.preprocessing import load_and_preprocess_data
from config import config

autoencoder_config = config["autoencoder"]
data_config = config["data"]
training_params = {
    "batch_size": autoencoder_config["batch_size"],
    "learning_rate": autoencoder_config["learning_rate"],
    "epochs": autoencoder_config["epochs"],
    "model_save_dir": config["paths"]["model_save_dir"]
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("📂 Loading and preprocessing data...")
X_train, X_val = load_and_preprocess_data()

train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32)), batch_size=config["batch_size"], shuffle=True)
val_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32)), batch_size=config["batch_size"], shuffle=False)

model = TimeSeriesAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=training_params["learning_rate"])

def train_autoencoder(model, train_loader, val_loader, criterion, optimizer, epochs):
    print("🚀 Starting training...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = batch[0].to(device)
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch[0].to(device)
                outputs = model(batch)
                val_loss += criterion(outputs, batch).item()

        print(
            f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss / len(train_loader):.6f} | Val Loss: {val_loss / len(val_loader):.6f}")

    print("✅ Training completed!")

train_autoencoder(model, train_loader, val_loader, criterion, optimizer, config["epochs"])

os.makedirs(config["model_save_dir"], exist_ok=True)
torch.save(model.state_dict(), os.path.join(config["model_save_dir"], "autoencoder.pth"))
print("💾 Model saved successfully!")
