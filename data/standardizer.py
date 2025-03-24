import os
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

INPUT_DIR = "data/processed"
OUTPUT_DIR = "data/standardized"
os.makedirs(OUTPUT_DIR, exist_ok=True)

csv_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.csv')]

def calculate_global_stats(csv_files, input_dir):
    global_stats = {
        "means": None,
        "stds": None
    }

    sums = None
    squares = None
    counts = 0

    for file_name in csv_files:
        file_path = os.path.join(input_dir, file_name)
        chunks = pd.read_csv(file_path, header=None, chunksize=100000)

        for chunk in chunks:
            if sums is None:
                sums = np.zeros(chunk.shape[1])
                squares = np.zeros(chunk.shape[1])

            sums += chunk.sum(axis=0)
            squares += (chunk ** 2).sum(axis=0)
            counts += len(chunk)

    global_stats["means"] = sums / counts
    global_stats["stds"] = np.sqrt((squares / counts) - (global_stats["means"] ** 2))

    return global_stats

def standardize_file(file_path, output_dir, means, stds):
    try:
        output_file = os.path.join(output_dir, os.path.basename(file_path))

        chunks = pd.read_csv(file_path, header=None, chunksize=100000)
        first_chunk = True

        for chunk in chunks:
            standardized_chunk = (chunk - means) / stds

            if first_chunk:
                standardized_chunk.to_csv(output_file, index=False, header=False, mode='w')
                first_chunk = False
            else:
                standardized_chunk.to_csv(output_file, index=False, header=False, mode='a')

        print(f"Standardized {file_path}")

    except Exception as e:
        print(f"Error standardizing {file_path}: {e}")

print("Calculating global statistics...")
global_stats = calculate_global_stats(csv_files, INPUT_DIR)
means = global_stats["means"]
stds = global_stats["stds"]

print("Global means:", means)
print("Global stds:", stds)

print("Standardizing files...")
with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
    futures = [executor.submit(standardize_file, os.path.join(INPUT_DIR, file_name), OUTPUT_DIR, means, stds) for file_name in csv_files]
    for future in as_completed(futures):
        future.result()

print("Standardization completed!")