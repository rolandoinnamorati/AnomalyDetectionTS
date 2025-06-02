import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from autoencoder import TimeSeriesAutoencoder
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import datetime
from sklearn.preprocessing import StandardScaler
import joblib

SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
model_save_path = SCRIPT_DIR / "autoencoder_model.pth"
preprocessed_data_path = SCRIPT_DIR / "preprocessed_data.pt"
scalers_save_path = SCRIPT_DIR / "feature_scalers.pkl"
data_dir = SCRIPT_DIR / "../data/windows" / "raw"
batch_size = 32
epochs = 10
date_column_index = 2
lr = 0.00001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = TimeSeriesAutoencoder().to(device)
criterion = nn.SmoothL1Loss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

def convert_to_timestamp(date_strings):
    timestamps = []
    for date_str in date_strings:
        try:
            dt_object = datetime.datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
            timestamp = dt_object.timestamp()
            timestamps.append(timestamp)
        except ValueError:
            timestamps.append(0)
    return np.array(timestamps, dtype=np.float32)

def preprocess_and_save_data(data_dir, date_column_index, preprocessed_data_path, scalers_save_path):
    print("Inizio pre-processing dei dati...")
    all_window_data = []
    all_features_data = []
    file_paths = list(data_dir.glob("*.npz"))

    for file_path in tqdm(file_paths, desc="Caricamento e pre-processing iniziale"):
        try:
            npzfile = np.load(file_path, allow_pickle=True)
            window_data = npzfile['data']
            features_data = npzfile['features']

            window_data[:, date_column_index] = convert_to_timestamp(window_data[:, date_column_index])

            window_data = window_data.reshape(-1, 200)
            features_data = features_data.reshape(-1, 4)

            all_window_data.append(window_data)
            all_features_data.append(features_data)

        except Exception as e:
            print(f"Errore durante il caricamento di {file_path}: {e}")
            continue

    if not all_window_data:
        raise ValueError("Nessun dato valido è stato caricato dai file NPZ.")

    full_window_data = np.concatenate(all_window_data, axis=0)
    full_features_data = np.concatenate(all_features_data, axis=0)

    print("Standardizzazione dei dati...")
    scalers = {}
    for i in tqdm(range(10), desc="Fitting scalers"):
        scaler = StandardScaler()
        full_window_data[:, i * 20: (i + 1) * 20] = scaler.fit_transform(
            full_window_data[:, i * 20: (i + 1) * 20]
        )
        scalers[f'scaler_{i}'] = scaler

    joblib.dump(scalers, scalers_save_path)
    print(f"Scalers salvati in {scalers_save_path}")
    final_data = np.concatenate((full_window_data, full_features_data), axis=1)
    np_array = np.array(final_data, dtype=np.float32)

    tensor_data = torch.tensor(np_array, dtype=torch.float32) # Non ancora .to(device)
    torch.save(tensor_data, preprocessed_data_path)
    print(f"Dati pre-processati salvati in {preprocessed_data_path}")
    return tensor_data

def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    for batch in dataloader:
        inputs = batch[0].to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, inputs)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss

# --- Main Training Logic ---
# Controlla se i dati pre-processati esistono già
if os.path.exists(preprocessed_data_path) and os.path.exists(scalers_save_path):
    print(f"Caricamento dati pre-processati da {preprocessed_data_path}...")
    tensor_data = torch.load(preprocessed_data_path)
    print(f"Dati caricati. Dimensione: {tensor_data.shape}")
else:
    print("Dati pre-processati non trovati, avvio il pre-processing.")
    tensor_data = preprocess_and_save_data(data_dir, date_column_index, preprocessed_data_path, scalers_save_path)

dataset = TensorDataset(tensor_data)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=os.environ['SLURM_CPUS_PER_TASK'], pin_memory=True) # Aggiunto num_workers e pin_memory

print("Inizio il training del modello...")
for epoch in range(epochs):
    epoch_loss = train_epoch(model, dataloader, optimizer)
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.6f}")

# Save Model
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, model_save_path)
print(f"Model saved to {model_save_path}")
