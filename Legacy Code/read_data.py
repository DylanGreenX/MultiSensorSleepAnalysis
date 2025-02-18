import os
import sqlite3
import pandas as pd

#This code exists to read in data from the proprietary AGD file format to a csv for future data modification. If you have raw data, disregard this file.


# Path to your AGD file
file_path = ""

# Connect to the SQLite database in the AGD file
if os.path.exists(file_path):
    # Connect to the SQLite database in the AGD file
    conn = sqlite3.connect(file_path)

    # Query the 'data' table and convert it into a pandas DataFrame
    data_df = pd.read_sql("SELECT * FROM data", conn)

    # Query the 'sleep' table and convert it into a pandas DataFrame
    sleep_df = pd.read_sql("SELECT * FROM sleep", conn)

    # Query the 'awakenings' table and convert it into a pandas DataFrame
    awakenings_df = pd.read_sql("SELECT * FROM awakenings", conn)

    # Close the connection after extraction
    conn.close()

    # Optionally, you can save the data to CSV files
    data_df.to_csv('data_table.csv', index=False)

    print("Data extracted and saved to CSV files!")
else:
    print(f"File not found: {file_path}")