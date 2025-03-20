import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

INPUT_DIR = "data/processed"

csv_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.csv')]

def calculate_statistics(file_path):
    try:
        chunks = pd.read_csv(file_path, header=None, chunksize=100000)
        stats = {
            "file_name": os.path.basename(file_path),
            "row_count": 0,
            "min_date": None,
            "max_date": None,
            "column_stats": {}
        }

        for chunk in chunks:
            stats["row_count"] += len(chunk)

            if len(chunk.columns) > 2:
                try:
                    chunk[2] = pd.to_datetime(chunk[2], errors='coerce')
                    chunk = chunk.dropna(subset=[2])
                    chunk_min_date = chunk[2].min()
                    chunk_max_date = chunk[2].max()

                    if stats["min_date"] is None or chunk_min_date < stats["min_date"]:
                        stats["min_date"] = chunk_min_date
                    if stats["max_date"] is None or chunk_max_date > stats["max_date"]:
                        stats["max_date"] = chunk_max_date
                except Exception as e:
                    print(f"Error processing timestamps in {file_path}: {e}")

            for i in range(3, len(chunk.columns)):
                column_name = f"column_{i}"
                if column_name not in stats["column_stats"]:
                    stats["column_stats"][column_name] = {
                        "min": float('inf'),
                        "max": float('-inf'),
                        "sum": 0,
                        "count": 0
                    }

                try:
                    chunk[i] = pd.to_numeric(chunk[i], errors='coerce')
                    chunk_cleaned = chunk.dropna(subset=[i])

                    column_min = chunk_cleaned[i].min()
                    column_max = chunk_cleaned[i].max()
                    column_sum = chunk_cleaned[i].sum()
                    column_count = len(chunk_cleaned[i])

                    if column_min < stats["column_stats"][column_name]["min"]:
                        stats["column_stats"][column_name]["min"] = column_min
                    if column_max > stats["column_stats"][column_name]["max"]:
                        stats["column_stats"][column_name]["max"] = column_max
                    stats["column_stats"][column_name]["sum"] += column_sum
                    stats["column_stats"][column_name]["count"] += column_count
                except Exception as e:
                    print(f"Error processing column {i} in {file_path}: {e}")

        for column_name, column_data in stats["column_stats"].items():
            if column_data["count"] > 0:
                column_data["mean"] = column_data["sum"] / column_data["count"]
            else:
                column_data["mean"] = None

        return stats

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        raise

def save_statistics(all_stats, output_file):
    stats_list = []
    for stats in all_stats:
        row = {
            "file_name": stats["file_name"],
            "row_count": stats["row_count"],
            "min_date": stats["min_date"],
            "max_date": stats["max_date"],
        }
        for column_name, column_data in stats["column_stats"].items():
            row[f"{column_name}_min"] = column_data["min"]
            row[f"{column_name}_max"] = column_data["max"]
            row[f"{column_name}_mean"] = column_data["mean"]
        stats_list.append(row)

    stats_df = pd.DataFrame(stats_list)
    stats_df.to_csv(output_file, index=False)
    print(f"Statistics saved to {output_file}")

def process_files_parallel(csv_files, input_dir, output_file):
    all_stats = []
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(calculate_statistics, os.path.join(input_dir, file_name)) for file_name in csv_files]
        for future in as_completed(futures):
            stats = future.result()
            all_stats.append(stats)
            print(f"Processed {stats['file_name']}")

    save_statistics(all_stats, output_file)

output_stats_file = os.path.join(INPUT_DIR, "statistics.csv")
process_files_parallel(csv_files, INPUT_DIR, output_stats_file)