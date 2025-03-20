import os

INPUT_DIR = "data/processed"

csv_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.csv')]

csv_files.sort(key=lambda x: int(x.split('.')[0]))

for index, old_name in enumerate(csv_files, start=1):
    old_path = os.path.join(INPUT_DIR, old_name)
    new_name = f"{index}.csv"
    new_path = os.path.join(INPUT_DIR, new_name)

    os.rename(old_path, new_path)
    print(f"Renamed {old_name} to {new_name}")

print("All files have been renamed consecutively.")