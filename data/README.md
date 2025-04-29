# 🚀 Dataset Process Pipeline

- **1**: elaborator.py (extracts data from mysql databases and inserts them into csv files vehicle by vehicle)
- **2**: rename_csv.py (reorder all csv files by renaming them in ascending order)
- **3**: stats_calculator.py (calculates statistics on the data contained in the csv files and generates a file containing them)
- **4**: order_csv.py (for each csv file it reorders all the data contained in ascending order)
- **5**: handle_null.py (fill the null values in the csv files)
- **6**: window_generator.py (processes csv files generating floating windows)
- **7**: window_classifier.py (classifies the windows generated in the previous step into anomalous or normal)
- **8**: standardizer.py