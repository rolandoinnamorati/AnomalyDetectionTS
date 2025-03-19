import os
from dotenv import load_dotenv
import pymysql.cursors
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

# Rimuovi le variabili d'ambiente esistenti per evitare conflitti
os.environ.pop("DATABASE_USER", None)
os.environ.pop("DATABASE_PASSWORD", None)
os.environ.pop("DATABASE_HOST", None)
os.environ.pop("DATABASE_NAME", None)
os.environ.pop("DATABASE_PORT", None)

# Carica le variabili d'ambiente dal file .env
load_dotenv()

# Definisci la directory di output e creala se non esiste
OUTPUT_DIR = "data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Definisci la dimensione del chunk
CHUNK_SIZE = 5000

# Funzione per creare una nuova connessione al database
def create_connection():
    return pymysql.connect(host=os.getenv('DATABASE_HOST'),
                          user=os.getenv('DATABASE_USER'),
                          password=os.getenv('DATABASE_PASSWORD'),
                          database=os.getenv('DATABASE_NAME'),
                          port=int(os.getenv('DATABASE_PORT')),
                          cursorclass=pymysql.cursors.DictCursor,
                          connect_timeout=60)

# Funzione per elaborare i dati di un singolo vehicle_id
def process_vehicle(vehicle_id):
    try:
        # Crea una nuova connessione per questo thread
        connection = create_connection()
        output_file = os.path.join(OUTPUT_DIR, f"{vehicle_id}.csv")
        query = f"SELECT * FROM gps_data_clean WHERE vehicle_id = {vehicle_id}"
        chunk_number = 0

        with connection.cursor() as cursor:
            cursor.execute(query)
            while True:
                rows = cursor.fetchmany(CHUNK_SIZE)
                if not rows:
                    break
                df = pd.DataFrame(rows)
                if chunk_number == 0:
                    df.to_csv(output_file, index=False, header=False, mode='w')
                else:
                    df.to_csv(output_file, index=False, header=False, mode='a')
                chunk_number += 1

        # Chiudi la connessione
        connection.close()
        return vehicle_id

    except Exception as e:
        print(f"Error processing vehicle_id {vehicle_id}: {e}")
        raise

try:
    # Connessione principale per ottenere i vehicle_id
    connection = create_connection()
    print("Connected to the database")

    # Query per contare il numero totale di righe nella tabella
    test_query = "SELECT COUNT(*) AS total FROM gps_data_clean"
    with connection.cursor() as cursor:
        cursor.execute(test_query)
        result = cursor.fetchone()
        print(f"Total rows in gps_data_clean: {result['total']}")

    # Query per ottenere tutti i vehicle_id distinti
    query_vehicle_ids = "SELECT DISTINCT vehicle_id FROM gps_data_clean WHERE vehicle_id > 299"
    with connection.cursor() as cursor:
        cursor.execute(query_vehicle_ids)
        vehicle_ids = cursor.fetchall()
        vehicle_ids = [row["vehicle_id"] for row in vehicle_ids]
        print(f"Found {len(vehicle_ids)} different vehicle ids")

    # Chiudi la connessione principale
    connection.close()

    # Usa ThreadPoolExecutor per elaborare i vehicle_id in parallelo
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(process_vehicle, vehicle_id) for vehicle_id in vehicle_ids]
        for future in as_completed(futures):
            print(f"Completed processing for vehicle ID {future.result()}")

except Exception as e:
    print(f"Error: {e}")