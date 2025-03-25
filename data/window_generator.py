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

(OUTPUT_DIR / "normal").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "anomalous").mkdir(parents=True, exist_ok=True)


def calculate_window_features(window):
    #TODO implementare il calcolo delle features di finestra
    #Traveled Distance
    #Speed Variation
    #Elapsed Time
    #Trajectory Change Rate ALTITUDE VARIATION IS BETTER?
    return np.array([0.0, 0.0, 0.0, 0.0])  # Placeholder


def process_chunk(chunk, file_stem, start_idx):
    windows_data = []
    for i in range(0, len(chunk) - WINDOW_SIZE + 1, STRIDE):
        window = chunk.iloc[i:i + WINDOW_SIZE].values
        features = calculate_window_features(window)
        label = "normal"
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
    csv_files = list(INPUT_DIR.glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files to process")

    with ThreadPoolExecutor(max_workers=THREADS) as executor:
        futures = [executor.submit(process_file, csv) for csv in csv_files]

        for future in tqdm(as_completed(futures), total=len(csv_files)):
            result = future.result()
            if "Error" in result:
                print(result)


if __name__ == "__main__":
    main()