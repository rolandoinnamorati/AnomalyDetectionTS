import csv
from datetime import datetime
import concurrent.futures
import os
from pathlib import Path
from tqdm import tqdm

INPUT_DIR = Path("data/processed")
OUTPUT_DIR = Path("data/ordered")
THREADS = os.cpu_count()

def write(csv_path, data):
    try:
        with open(csv_path, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            for row in data:
                row[2] = row[2].isoformat(timespec='microseconds').replace('+00:00', 'Z')
                writer.writerow(row)
        return f"File written successfully: {csv_path}"
    except Exception as e:
        return f"Error writing file {csv_path}: {str(e)}"

def read_and_order(csv_path):
    data = []
    try:
        with open(csv_path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) >= 3:
                    try:
                        row[2] = datetime.fromisoformat(row[2].replace('Z', '+00:00'))
                        data.append(row)
                    except ValueError:
                        print(f"Invalid date format in row: {row}")
                else:
                    print(f"Row has insufficient columns: {row}")
    except FileNotFoundError:
        return f"File not found: {csv_path}"
    except Exception as e:
        return f"Error reading file {csv_path}: {str(e)}"

    data.sort(key=lambda row: row[2])
    return data

def process_file(csv_path):
    file_stem = csv_path.stem
    try:
        ordered = read_and_order(csv_path)
        if(isinstance(ordered, list)):
            result = write(OUTPUT_DIR / f"{file_stem}.csv", ordered)
            return result
        else:
            return ordered
    except Exception as e:
        return f"Error in {file_stem}: {str(e)}"

def main():
    csv_files = list(INPUT_DIR.glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files to process")

    with concurrent.futures.ThreadPoolExecutor(max_workers=THREADS) as executor:
        futures = [executor.submit(process_file, csv) for csv in csv_files]

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(csv_files)):
            result = future.result()
            if "Error" in result:
                print(result)

if __name__ == "__main__":
    main()