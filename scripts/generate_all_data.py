import pandas as pd
import numpy as np
import sqlite3
import os
from datetime import datetime, timedelta

# --- CONFIGURATION ---
# DYNAMIC DATES: Always relative to today's execution
NOW = datetime.now()
START_DATE = NOW - timedelta(days=365*2) # 2 Years of history
CURRENT_DATE = NOW 
DAYS_HISTORY = (CURRENT_DATE - START_DATE).days
DAYS_FUTURE = 45 # Forecast 45 days ahead to be safe

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "data.db")
CSV_PATH = os.path.join(BASE_DIR, "data", "revenue_train.csv")

NUM_USERS = 5

def generate_and_seed():
    print(f"--- 1. Generating Training Data (CSV) ---")
    print(f"    Timeline: {START_DATE.date()} to {CURRENT_DATE.date()}")
    
    end_history = START_DATE + timedelta(days=DAYS_HISTORY)
    dates = pd.date_range(start=START_DATE, end=end_history, freq='D')
    
    daily_records = []
    
    for d in dates:
        # Seasonality logic
        season_factor = 1 + 0.3 * np.sin((d.dayofyear / 365) * 2 * np.pi - 0.5)
        total_load = 500 + (season_factor * 200) + np.random.normal(0, 50)
        rate = 0.15 + (0.05 if d.month in [6,7,8] else 0)
        revenue = total_load * rate
        
        daily_records.append({
            "date": d.strftime("%Y-%m-%d"),
            "total_load": round(total_load, 2),
            "revenue": round(revenue, 2)
        })

    if not os.path.exists(os.path.dirname(CSV_PATH)): os.makedirs(os.path.dirname(CSV_PATH))
    pd.DataFrame(daily_records).to_csv(CSV_PATH, index=False)
    print(f"    Saved {len(daily_records)} rows to {CSV_PATH}")

    print(f"--- 2. Populating Database (SQLite) ---")
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS users")
    cur.execute("DROP TABLE IF EXISTS meter_loads")
    cur.execute("CREATE TABLE users (username TEXT PRIMARY KEY, meter_id INTEGER)")
    cur.execute("CREATE TABLE meter_loads (meter_id INTEGER, date_time TEXT, forecasted_load REAL)")
    
    users = [(f"user_{i}", 1000+i) for i in range(1, NUM_USERS+1)]
    cur.executemany("INSERT INTO users VALUES (?,?)", users)
    
    # Generate Hourly Data: History + Future
    end_future = end_history + timedelta(days=DAYS_FUTURE)
    print(f"    Generating hourly forecasts up to: {end_future.date()}")
    
    all_dates = pd.date_range(start=START_DATE, end=end_future, freq='h')
    
    # Future targets
    future_dates = pd.date_range(start=end_history + timedelta(days=1), end=end_future, freq='D')
    future_targets = {}
    for d in future_dates:
        season_factor = 1 + 0.3 * np.sin((d.dayofyear / 365) * 2 * np.pi - 0.5)
        future_targets[d.strftime("%Y-%m-%d")] = 500 + (season_factor * 200)

    history_targets = {row['date']: row['total_load'] for row in daily_records}
    daily_targets = {**history_targets, **future_targets}
    
    db_rows = []
    batch_size = 10000
    
    for ts in all_dates:
        date_key = ts.strftime("%Y-%m-%d")
        if date_key not in daily_targets: continue
        
        daily_total = daily_targets[date_key]
        hour_factor = 1 + 0.5 * np.sin((ts.hour - 6) * np.pi / 12)
        hourly_system_load = (daily_total / 24) * hour_factor
        
        for i in range(NUM_USERS):
            meter_id = 1000 + (i + 1)
            user_load = (hourly_system_load / NUM_USERS) * np.random.uniform(0.8, 1.2)
            db_rows.append((meter_id, ts.strftime("%Y-%m-%d %H:%M:%S"), round(user_load, 2)))
        
        if len(db_rows) >= batch_size:
            cur.executemany("INSERT INTO meter_loads VALUES (?,?,?)", db_rows)
            db_rows = []

    if db_rows:
        cur.executemany("INSERT INTO meter_loads VALUES (?,?,?)", db_rows)
        
    conn.commit()
    conn.close()
    print("✅ Database population complete.")

if __name__ == "__main__":
    generate_and_seed()