import sqlite3
import pandas as pd
import os

# Robust path handling
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "data.db")

def create_db_if_missing():
    """Creates the table structure if DB doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, meter_id INTEGER)")
        conn.execute("CREATE TABLE IF NOT EXISTS meter_loads (meter_id INTEGER, date_time TEXT, forecasted_load REAL)")
        conn.commit()
    finally:
        conn.close()

def run_select_query(sql):
    """Executes a read-only SQL query and returns a DataFrame."""
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(sql, conn)
        return df
    except Exception as e:
        raise e
    finally:
        conn.close()