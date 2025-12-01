#!/usr/bin/env python3
"""
BeAST Synthetic Data Playback
Reads the 12-hour synthetic dataset and inserts it into the database
at simulated real-time speed or faster.
"""

import json
import time
import os
import sqlite3
from datetime import datetime
from pathlib import Path

# Configuration
SYNTHETIC_DATA = Path(__file__).parent.parent / "TheBeast" / "TheBeast" / "beast_sensordata" / "Synthetic Data" / "earpiece_12hour_full_session.jsonl"
DB_PATH = Path(__file__).parent.parent / "TheBeast" / "TheBeast" / "beast" / "data" / "beast_local.db"
PLAYBACK_SPEED = 10.0  # 10x speed (1 second real time = 10 seconds of data)

print(f"BeAST Synthetic Data Playback")
print(f"Data file: {SYNTHETIC_DATA}")
print(f"Database: {DB_PATH}")
print(f"Playback speed: {PLAYBACK_SPEED}x")
print()

# Create database tables if they don't exist
db_conn = sqlite3.connect(DB_PATH)
cursor = db_conn.cursor()

# Create sensors table (simple version for now)
cursor.execute("""
    CREATE TABLE IF NOT EXISTS sensor_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        side TEXT,
        mode TEXT,
        activity TEXT,
        sensor_type TEXT,
        sensor_status TEXT,
        sensor_freq REAL,
        sensor_data TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
""")
db_conn.commit()

print("Database tables ready")
print("Starting playback...")
print()

# Read and play back data
with open(SYNTHETIC_DATA, 'r') as f:
    last_time = None
    line_count = 0
    
    for line in f:
        line_count += 1
        
        try:
            data = json.loads(line.strip())
            
            # Extract timestamp
            time_array = data['time']  # [year, month, day, hour, minute, second]
            current_time = datetime(*time_array)
            
            # Calculate delay based on time difference
            if last_time is not None:
                time_diff = (current_time - last_time).total_seconds()
                sleep_time = time_diff / PLAYBACK_SPEED
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            last_time = current_time
            
            # Insert each sensor reading
            for device in data.get('devices', []):
                cursor.execute("""
                    INSERT INTO sensor_data (
                        timestamp, side, mode, activity,
                        sensor_type, sensor_status, sensor_freq, sensor_data
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    current_time.isoformat(),
                    data['side'],
                    data['mode'],
                    data['activity'],
                    device['name'],
                    device['status'],
                    device['freq'],
                    json.dumps(device['data'])
                ))
            
            # Commit every 100 lines
            if line_count % 100 == 0:
                db_conn.commit()
                print(f"Processed {line_count} lines... [{current_time.strftime('%H:%M:%S')}]")
        
        except Exception as e:
            print(f"Error processing line {line_count}: {e}")
            continue

# Final commit
db_conn.commit()
db_conn.close()

print()
print(f"Playback complete! Processed {line_count} sensor readings.")
