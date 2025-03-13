from decouple import config as ENV
import pymysql.cursors
import pandas as pd

connection = pymysql.connect(host=ENV('database_host'),
                                 user=ENV('database_user'),
                                 password=ENV('database_password'),
                                 database=ENV('database_name'),
                                 cursorclass=pymysql.cursors.DictCursor)

query = "SELECT vehicle_id,latitude,longitude,date,speed,altitude,satellites,gsm_signal,external_supply_volt,internal_supply_volt,remaining_battery FROM gps_data_clean"
df = pd.read_sql(query, connection)
connection.close()

df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by=['vehicle_id', 'date'])