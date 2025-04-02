import os
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm

INPUT_DIR = Path("data/processed")
OUTPUT_DIR = Path("data/windows")
WINDOW_SIZE = 20
STRIDE = 5
CHUNKSIZE = 100000
THREADS = os.cpu_count()

(OUTPUT_DIR / "raw").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "normal").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "anomalous").mkdir(parents=True, exist_ok=True)

def calculate_window_features(window):
    """
        Calcola le 4 metriche per una finestra 20x10.
        Input: window (np.array 20x10) con colonne:
            0: latitudine, 1: longitudine, 2: timestamp, 3: velocità,
            4: altitudine, 5: satelliti, 6: segnale GSM,
            7: volt_est, 8: volt_int, 9: batteria
        Output: np.array [distanza, variazione_velocità, tempo, variazione_altitudine]
        """
    # 1. Distanza totale (metri) - Uso formula di Haversine
    # lat_lon = np.radians(window[:, :2])  # Converti in radianti
    # lat_diff = np.diff(lat_lon[:, 0])
    # lon_diff = np.diff(lat_lon[:, 1])
    # a = (np.sin(lat_diff / 2) ** 2 +
    #      np.cos(lat_lon[:-1, 0]) * np.cos(lat_lon[1:, 0]) * np.sin(lon_diff / 2) ** 2)
    # distanze = 6371e3 * 2 * np.arcsin(np.sqrt(a))  # Raggio terrestre in metri
    # distanza_totale = np.sum(distanze)
    distanza_totale = 0

    # 2. Variazione di velocità (m/s²)
    # velocita = window[:, 3] * 0.277778  # Converti km/h → m/s
    # accelerazioni = np.diff(velocita) / np.diff(window[:, 2])  # Derivata prima
    # variazione_velocita = np.mean(np.abs(accelerazioni))
    variazione_velocita = 0

    # 3. Tempo trascorso (secondi)
    timestamps = np.array([
        pd.Timestamp(t) if isinstance(t, str) and len(t) == 19
        else pd.NaT
        for t in window[:, 2]
    ])
    tempo_trascorso = (timestamps[-1] - timestamps[0]).total_seconds() if pd.notna(timestamps).all() else 0.0

    # 4. Variazione altitudine (metri)
    altitudini = window[:, 4]
    variazione_altitudine = np.max(altitudini) - np.min(altitudini)

    return np.array([distanza_totale, variazione_velocita, tempo_trascorso, variazione_altitudine])


def process_chunk(chunk, file_stem, start_idx):
    windows_data = []
    for i in range(0, len(chunk) - WINDOW_SIZE + 1, STRIDE):
        window = chunk.iloc[i:i + WINDOW_SIZE].values
        features = calculate_window_features(window)
        label = "raw"
        windows_data.append((window, features, label, i + start_idx))
    return windows_data


def process_file(csv_path):
    file_stem = csv_path.stem
    try:
        for chunk_idx, chunk in enumerate(pd.read_csv(csv_path, header=None, chunksize=CHUNKSIZE)):
            start_idx = chunk_idx * CHUNKSIZE
            windows = process_chunk(chunk, file_stem, start_idx)

            for window, features, label, window_idx in windows:
                np.savez(
                    OUTPUT_DIR / label / f"{file_stem}_window_{window_idx}.npz",
                    data=window,
                    features=features
                )
        return f"OK: {file_stem}"
    except Exception as e:
        return f"Error in {file_stem}: {str(e)}"


def main():
    # csv_files = list(INPUT_DIR.glob("*.csv"))
    csv_files = list(INPUT_DIR.glob("10.csv"))
    print(f"Found {len(csv_files)} CSV files to process")

    with ThreadPoolExecutor(max_workers=THREADS) as executor:
        futures = [executor.submit(process_file, csv) for csv in csv_files]

        for future in tqdm(as_completed(futures), total=len(csv_files)):
            result = future.result()
            if "Error" in result:
                print(result)


if __name__ == "__main__":
    main()