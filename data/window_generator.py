import os
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))
from utils import haversine

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
        Input: window (np.array 20x10)
        Output: np.array [distanza, variazione_velocità, tempo, variazione_altitudine]
        """
    # 1. Distanza totale (metri)
    coords = [(row[0], row[1]) for row in window]
    distanza_totale = 0.0
    for i in range(len(coords) - 1):
        lat1, lon1 = coords[i]
        lat2, lon2 = coords[i + 1]
        distanza_totale += haversine(lat1, lon1, lat2, lon2)

    # 2. Variazione di velocità (km/h)
    velocita = [row[3] for row in window]
    velocita_max = max(velocita)
    velocita_min = min(velocita)
    variazione_velocita = velocita_max - velocita_min

    # 3. Tempo trascorso (secondi)
    dates = [datetime.strptime(row[2], '%Y-%m-%d %H:%M:%S') for row in window]
    min_date = min(dates)
    max_date = max(dates)
    tempo_trascorso = (max_date - min_date).total_seconds()

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