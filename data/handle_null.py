import csv
from datetime import datetime
import concurrent.futures
import os
from pathlib import Path
from tqdm import tqdm

THREADS = os.cpu_count()

def process_csv_file(csv_path):
    file_stem = csv_path.stem
    try:
        updated_rows = []
        previous_row_data = None

        with open(csv_path, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader, None)
            if header:
                updated_rows.append(header[:10])
            for row in reader:
                if len(row) >= 3:
                    current_row = list(row[:10])

                    try:
                        current_row[2] = datetime.fromisoformat(current_row[2].replace('Z', '+00:00'))

                        for i in range(3, 9):
                            if i < len(current_row) and current_row[i] == '':
                                if previous_row_data and len(previous_row_data) > i:
                                    current_row[i] = previous_row_data[i]
                                else:
                                    current_row[i] = '0.0'

                        # Handle the battery field at index 9
                        if len(current_row) < 10 or current_row[9] == '':
                            if previous_row_data and len(previous_row_data) >= 10 and previous_row_data[9] != '':
                                if len(current_row) < 10:
                                    current_row.append(previous_row_data[9])
                                else:
                                    current_row[9] = previous_row_data[9]
                            else:
                                if len(current_row) < 10:
                                    current_row.append('100.0')
                                else:
                                    current_row[9] = '100.0'

                        updated_rows.append(current_row)
                        previous_row_data = list(current_row)

                    except ValueError:
                        print(f"Skipping row in '{csv_path.name}' due to invalid datetime format: {row}")
                        updated_rows.append(row[:10])
                else:
                    print(f"Skipping row in '{csv_path.name}' as it has fewer than 3 columns: {row}")
                    updated_rows.append(row[:10] if len(row) >= 10 else row)

        updated_rows.sort(key=lambda x: x[2] if isinstance(x[2], datetime) else datetime.min)

        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(updated_rows)

        return f"Processed and updated: {csv_path.name}"

    except FileNotFoundError:
        return f"Error: File not found: {csv_path}"
    except Exception as e:
        return f"Error processing {file_stem}: {str(e)}"

def main():
    input_dir = Path("data/ordered")
    csv_files = list(input_dir.glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files to process and update in place")

    with concurrent.futures.ThreadPoolExecutor(max_workers=THREADS) as executor:
        futures = [executor.submit(process_csv_file, csv) for csv in csv_files]

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(csv_files)):
            result = future.result()
            print(result)

if __name__ == "__main__":
    main()