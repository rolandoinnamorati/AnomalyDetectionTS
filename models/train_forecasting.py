import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from forecasting import TimeSeriesForecaster
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import datetime
from sklearn.preprocessing import StandardScaler
import joblib

SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
model_save_path = SCRIPT_DIR / "forecaster_model.pth"
preprocessed_data_path = SCRIPT_DIR / "preprocessed_data.pt"
scalers_save_path = SCRIPT_DIR / "feature_scalers.pkl"
data_dir = SCRIPT_DIR / "../data/windows" / "raw"  # Usato solo se i dati pre-processati non esistono

batch_size = 32
epochs = 100
date_column_index = 2
lr = 0.0001

window_size = 20
num_features_per_timestep = 10
num_global_features = 4
total_input_dim = 200 + num_global_features

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = TimeSeriesForecaster().to(device)
criterion = nn.HuberLoss(delta=1.0)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

print(f"Forecasting Model Architecture:\n{model}")

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
    print("Inizio pre-processing dei dati (se non già presenti)...")
    all_window_data = []
    all_features_data = []
    file_paths = list(data_dir.glob("*.npz"))

    for file_path in tqdm(file_paths, desc="Caricamento e pre-processing iniziale"):
        try:
            npzfile = np.load(file_path, allow_pickle=True)
            window_data = npzfile['data']
            features_data = npzfile['features']

            window_data[:, date_column_index] = convert_to_timestamp(window_data[:, date_column_index])
            window_data = window_data.astype(np.float32)

            window_data_reshaped = window_data.reshape(-1, 200)
            features_data_reshaped = features_data.reshape(-1, 4)

            all_window_data.append(window_data_reshaped)
            all_features_data.append(features_data_reshaped)

        except Exception as e:
            print(f"Errore durante il caricamento di {file_path}: {e}")
            continue

    if not all_window_data:
        raise ValueError("Nessun dato valido è stato caricato dai file NPZ.")

    full_window_data_all_features_200 = np.concatenate(all_window_data, axis=0)  # Tutti i dati della window (200 features)
    full_global_features_4 = np.concatenate(all_features_data, axis=0)  # Tutte le features globali (4 features)

    print("Standardizzazione dei dati...")
    scalers = {}
    for i in tqdm(range(num_features_per_timestep), desc="Fitting scalers"):  # Sono 10 scaler, uno per ogni feature originale
        scaler = StandardScaler()
        full_window_data_all_features_200[:, i * window_size: (i + 1) * window_size] = scaler.fit_transform(
            full_window_data_all_features_200[:, i * window_size: (i + 1) * window_size]
        )
        scalers[f'scaler_{i}'] = scaler

    joblib.dump(scalers, scalers_save_path)  # Sovrascriverà gli scaler dell'autoencoder se sono lo stesso file
    print(f"Scalers salvati in {scalers_save_path}")

    # Salva il tensore completo da 204 per consistenza con il loading, ma poi lo divideremo
    final_combined_data = np.concatenate((full_window_data_all_features_200, full_global_features_4), axis=1)
    tensor_data = torch.tensor(final_combined_data, dtype=torch.float32)
    torch.save(tensor_data, preprocessed_data_path)
    print(f"Dati pre-processati salvati in {preprocessed_data_path}")
    return tensor_data


def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    for batch_data in dataloader:
        # batch_data[0] è il tensore da 204 features per N campioni
        full_inputs = batch_data[0].to(device)

        # Divide l'input nelle due parti: time_series (20x10) e global (4)
        x_time_series_flat = full_inputs[:, :200]
        x_global = full_inputs[:, 200:]

        x_time_series = x_time_series_flat.view(-1, window_size, num_features_per_timestep)

        # Target semplificato: le 10 features dell'ultimo timestep dell'input corrente
        y_true = x_time_series[:, -1, :]  # Questo è (batch_size, 10)

        # Forward pass
        outputs = model(x_time_series, x_global)  # Il modello prevede 10 features

        # Calcolo della loss
        loss = criterion(outputs, y_true)  # Confronta previsione con target

        # Backward pass e ottimizzazione
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


# --- Main Training Logic ---
if os.path.exists(preprocessed_data_path) and os.path.exists(scalers_save_path):
    print(f"Caricamento dati pre-processati da {preprocessed_data_path}...")
    tensor_data = torch.load(preprocessed_data_path)
    print(f"Dati caricati. Dimensione: {tensor_data.shape}")
else:
    print("Dati pre-processati non trovati, avvio il pre-processing.")
    tensor_data = preprocess_and_save_data(data_dir, date_column_index, preprocessed_data_path, scalers_save_path)

dataset = TensorDataset(tensor_data)
num_workers_val = 0
if 'SLURM_CPUS_PER_TASK' in os.environ:
    try:
        num_workers_val = int(os.environ['SLURM_CPUS_PER_TASK'])
    except ValueError:
        print("Warning: SLURM_CPUS_PER_TASK non è un numero valido. Usando num_workers=0.")
        num_workers_val = 0
else:
    num_workers_val = os.cpu_count()
    print(f"Usando num_workers={num_workers_val} (non-SLURM environment or SLURM_CPUS_PER_TASK not set).")

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers_val, pin_memory=True)

print("\nInizio il training del modello di Forecasting...")
for epoch in range(epochs):
    epoch_loss = train_epoch(model, dataloader, optimizer, criterion)
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.6f}")

torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, model_save_path)
print(f"Model saved to {model_save_path}")