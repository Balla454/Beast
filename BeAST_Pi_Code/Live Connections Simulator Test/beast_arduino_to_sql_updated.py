import serial
import json
import uuid
from datetime import datetime
from pathlib import Path
import time
import os
import psycopg2

# ----------------------------------------------
# CONFIGURATION
# ----------------------------------------------

# For the Pi:
SERIAL_PORT = "/dev/ttyACM0"   # UNO R4 WiFi shows up here
BAUD_RATE = 115200

# Output folder where session SQL will be saved
OUTPUT_DIR = Path("./beast_session_exports")
OUTPUT_DIR.mkdir(exist_ok=True)

# ----------------------------------------------
# DATABASE CONFIG (Antonio's beast schema)
# ----------------------------------------------
DB_HOST = os.getenv("BEAST_DB_HOST", "localhost")
DB_PORT = int(os.getenv("BEAST_DB_PORT", "5432"))
DB_NAME = os.getenv("BEAST_DB_NAME", "beast")
DB_USER = os.getenv("BEAST_DB_USER", "beast")
DB_PASS = os.getenv("BEAST_DB_PASSWORD", "beast")  # default password for beast user


def wipe_tables():
    """
    Wipe raw tables at the start of a new session so each run
    is a clean slate in the database.
    """
    print("Connecting to PostgreSQL to wipe raw tables...")
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
    )
    cur = conn.cursor()

    # You can add/remove tables here as needed
    cur.execute(
        """
        TRUNCATE
            raw.cognitive_metrics,
            raw.ppg_reading,
            raw.earpiece_data,
            raw.eeg_band
        RESTART IDENTITY;
        """
    )

    conn.commit()
    cur.close()
    conn.close()
    print("✅ Raw tables wiped for new session.")


# ----------------------------------------------
# TABLE MAPPINGS (ANTONIO'S SCHEMA)
# ----------------------------------------------
# JSON "type"  ->  PostgreSQL table in beast_schema.sql
TABLES = {
    "eeg_raw":         "raw.earpiece_data",       # raw EEG
    "eeg_bands":       "raw.eeg_band",           # alpha/beta/gamma/delta/theta
    "clinical_metrics":"raw.cognitive_metrics",   # cognitive_load/alertness/fatigue
    "hydration":       "raw.hydration_reading",   # hydration_pct
    "ppg":             "raw.ppg_reading",         # hr_bpm / spo2_pct
    "core_temp":       "raw.temp_reading",        # core_temp_c
    "ambient_temp":    "raw.environment",         # ambient_temp
    # "impact":        "raw.bioz_reading",        # IMU – only map if packets have accel/gyro
}

# ----------------------------------------------
# HELPERS
# ----------------------------------------------

def new_ids():
    session_id = str(uuid.uuid4())
    device_id = str(uuid.uuid4())
    return session_id, device_id

def pg_literal(value):
    """Converts Python values into SQL-safe Postgres literals."""
    if value is None:
        return "NULL"

    if isinstance(value, (int, float)):
        return str(value)

    if isinstance(value, list):
        inside = ", ".join(pg_literal(v) for v in value)
        return f"ARRAY[{inside}]"

    if isinstance(value, dict):
        s = json.dumps(value)
        s = s.replace("'", "''")
        return f"'{s}'::jsonb"

    # strings
    s = str(value).replace("'", "''")
    return f"'{s}'"

def build_insert(pkt, session_id, device_id):
    """
    Build INSERT INTO ... statement for a single packet,
    mapped to Antonio's beast_schema.sql tables.
    """
    ptype = pkt.get("type")
    if ptype not in TABLES:
        return None

    table = TABLES[ptype]

    # Timestamp – assume packet has ISO ts or fallback to "now"
    ts = pkt.get("ts") or datetime.utcnow().isoformat()

    # --------- PACKET-SPECIFIC MAPPINGS ---------
    if ptype == "eeg_raw":
        # Maps to raw.earpiece_data:
        #   ts, device_id, session_id, side, mode, data[]
        side = pkt.get("side", "L")           # or derive from channel if you prefer
        mode = pkt.get("mode", "LIVE")
        samples = pkt.get("samples")          # expect list of numbers

        cols = ["ts", "device_id", "session_id", "side", "mode", "data"]
        vals = [
            pg_literal(ts),
            pg_literal(device_id),
            pg_literal(session_id),
            pg_literal(side),
            pg_literal(mode),
            pg_literal(samples),
        ]

    elif ptype == "eeg_bands":
        # Maps to raw.eeg_band:
        #   ts, device_id, session_id, alpha, beta, gamma, delta, theta
        cols = [
            "ts", "device_id", "session_id",
            "alpha", "beta", "gamma", "delta", "theta",
        ]
        vals = [
            pg_literal(ts),
            pg_literal(device_id),
            pg_literal(session_id),
            pg_literal(pkt.get("alpha")),
            pg_literal(pkt.get("beta")),
            pg_literal(pkt.get("gamma")),
            pg_literal(pkt.get("delta")),
            pg_literal(pkt.get("theta")),
        ]

    elif ptype == "clinical_metrics":
        # Maps to raw.cognitive_metrics:
        #   ts, device_id, session_id, cognitive_load, alertness, fatigue
        cols = [
            "ts", "device_id", "session_id",
            "cognitive_load", "alertness", "fatigue",
        ]
        vals = [
            pg_literal(ts),
            pg_literal(device_id),
            pg_literal(session_id),
            pg_literal(pkt.get("cognitive_load")),
            pg_literal(pkt.get("alertness")),
            pg_literal(pkt.get("fatigue")),
        ]
        # NOTE: zone_durations from the packet is ignored here and
        # could be handled by a separate aggregator -> agg.zone_durations.

    elif ptype == "hydration":
        # Maps to raw.hydration_reading:
        #   ts, device_id, session_id, hydration_pct
        cols = ["ts", "device_id", "session_id", "hydration_pct"]
        vals = [
            pg_literal(ts),
            pg_literal(device_id),
            pg_literal(session_id),
            pg_literal(pkt.get("value")),     # assuming "value" is percentage
        ]

    elif ptype == "ppg":
        # Maps to raw.ppg_reading:
        #   ts, device_id, session_id, hr_bpm, spo2_pct, raw_data[]
        cols = ["ts", "device_id", "session_id", "hr_bpm", "spo2_pct", "raw_data"]
        vals = [
            pg_literal(ts),
            pg_literal(device_id),
            pg_literal(session_id),
            pg_literal(pkt.get("heart_rate")),
            pg_literal(pkt.get("spo2")),
            pg_literal(pkt.get("raw_data")),  # or NULL if not present
        ]
        # hr_zone_durations / spo2_zone_durations would feed agg tables later.

    elif ptype == "core_temp":
        # Maps to raw.temp_reading:
        #   ts, device_id, session_id, core_temp_c
        cols = ["ts", "device_id", "session_id", "core_temp_c"]
        vals = [
            pg_literal(ts),
            pg_literal(device_id),
            pg_literal(session_id),
            pg_literal(pkt.get("temperature_c")),
        ]

    elif ptype == "ambient_temp":
        # Maps to raw.environment:
        #   ts, device_id, ambient_temp, humidity_pct, heat_index_c
        cols = ["ts", "device_id", "ambient_temp", "humidity_pct", "heat_index_c"]
        vals = [
            pg_literal(ts),
            pg_literal(device_id),
            pg_literal(pkt.get("temperature_c")),
            pg_literal(pkt.get("humidity_pct")),
            pg_literal(pkt.get("heat_index_c")),
        ]

    else:
        # If you later want to handle "impact" into raw.bioz_reading,
        # you'd add another elif here with accel/gyro mapping.
        return None

    col_sql = ", ".join(cols)
    val_sql = ", ".join(vals)
    return f"INSERT INTO {table} ({col_sql}) VALUES ({val_sql});"

# ----------------------------------------------
# MAIN
# ----------------------------------------------

def main():

    # 1) Wipe DB tables for a fresh session
    wipe_tables()

    # 2) Connect to PostgreSQL for live inserts
    print("Connecting to PostgreSQL for live inserts...")
    pg_conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
    )
    pg_conn.autocommit = True
    pg_cursor = pg_conn.cursor()
    print("✅ PostgreSQL connected for live inserts")

    # 3) Connect to Arduino and start live stream
    print(f"Connecting to Arduino on {SERIAL_PORT} @ {BAUD_RATE} baud...")
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)

    # Give the Arduino a moment to reset after opening serial
    time.sleep(2.0)

    # Send single 'L' command to start full live stream
    print("Sending 'L' command to Arduino to start live stream...")
    ser.write(b"L\n")
    ser.flush()

    session_id, device_id = new_ids()
    insert_count = 0

    outfile_name = f"session_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.sql"
    outfile_path = OUTPUT_DIR / outfile_name

    with outfile_path.open("w", encoding="utf-8") as f:
        f.write("-- BeAST End-of-Session Export (Antonio schema)\n")
        f.write(f"-- session_id: {session_id}\n")
        f.write(f"-- device_id:  {device_id}\n\n")
        f.write("BEGIN;\n\n")

        print(f"Writing session SQL to: {outfile_path}")
        print("Press CTRL+C to end the session.\n")

        try:
            while True:
                line = ser.readline().decode(errors="ignore").strip()
                if not line:
                    continue

                # Expect JSON line from Arduino
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Tanya's firmware: uses a 'data_batch' wrapper for multiple packets
                if msg.get("type") == "data_batch":
                    for pkt in msg.get("batch", []):
                        insert_sql = build_insert(pkt, session_id, device_id)
                        if insert_sql:
                            f.write(insert_sql + "\n")
                            # Also insert directly into PostgreSQL
                            try:
                                pg_cursor.execute(insert_sql)
                                insert_count += 1
                                if insert_count % 10 == 0:
                                    print(f"✅ Inserted {insert_count} records into PostgreSQL")
                            except Exception as e:
                                print(f"⚠️ DB insert error: {e}")

        except KeyboardInterrupt:
            print(f"\nSession ended by user. Total inserts: {insert_count}\n")

        f.write("\nCOMMIT;\n")

    pg_cursor.close()
    pg_conn.close()
    print(f"Session SQL saved: {outfile_path}")

if __name__ == "__main__":
    main()
