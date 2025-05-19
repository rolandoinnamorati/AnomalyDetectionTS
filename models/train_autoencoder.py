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
from sklearn.preprocessing import MinMaxScaler, StandardScaler

SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
model_save_path = "autoencoder_model.pth"
data_dir = SCRIPT_DIR / "../data/windows" / "raw"
batch_size = 32
epochs = 10
date_column_index = 2
lr = 0.00001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = TimeSeriesAutoencoder().to(device)
# criterion = nn.MSELoss()
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
            print(f"Impossibile convertire la data: {date_str}. Usando 0.")
            timestamps.append(0)
    return np.array(timestamps, dtype=np.float32)

def train_epoch(model, dataloader, optimizer, criterion, epoch):
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

def process_data_files(data_dir, model, optimizer, criterion, epochs):
    file_paths = list(data_dir.glob("*.npz"))
    for epoch in range(epochs):
        print(f"Inizio Epoca {epoch + 1}")
        epoch_loss = 0.0
        for file_path in tqdm(file_paths, desc="Processing Files"):
            try:
                npzfile = np.load(file_path, allow_pickle=True)
                window_data = npzfile['data']
                features_data = npzfile['features']

                window_data[:, date_column_index] = convert_to_timestamp(window_data[:, date_column_index])

                window_data = window_data.reshape(-1, 200)
                features_data = features_data.reshape(-1, 4)

                # Standardizza le 10 features di window_data separatamente
                for i in range(10):
                    scaler = StandardScaler()
                    window_data[:, i * 20: (i + 1) * 20] = scaler.fit_transform(
                        window_data[:, i * 20: (i + 1) * 20]
                    )

                window_data = np.concatenate((window_data, features_data), axis=1)

                np_array = np.array(window_data, dtype=np.float32)

                tensor_data = torch.tensor(np_array, dtype=torch.float32).to(device)
                dataset = TensorDataset(tensor_data)
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
                epoch_loss += train_epoch(model, dataloader, optimizer, criterion, epoch)

            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        print(f"Epoch {epoch + 1} Total Loss: {epoch_loss}")

process_data_files(data_dir, model, optimizer, criterion, epochs)

# Save Model
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, model_save_path)
print(f"Model saved to {model_save_path}")
