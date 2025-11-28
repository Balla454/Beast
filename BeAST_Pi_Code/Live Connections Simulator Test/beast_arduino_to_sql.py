import serial
import json
import uuid
from datetime import datetime
from pathlib import Path

# ----------------------------------------------
# CONFIGURATION
# ----------------------------------------------

# CHANGE THIS depending on where you're running it:
# Windows example: SERIAL_PORT = "COM7"
# Pi example:      SERIAL_PORT = "/dev/ttyACM0"
SERIAL_PORT = "COM7"
BAUD_RATE = 115200

# Output folder where session SQL will be saved
OUTPUT_DIR = Path("./beast_session_exports")
OUTPUT_DIR.mkdir(exist_ok=True)


# Mapping packet types ➜ table names
TABLES = {
    "eeg_raw": "raw.eeg_raw",
    "eeg_bands": "raw.eeg_bands",
    "clinical_metrics": "raw.clinical_metrics",
    "hydration": "raw.hydration",
    "ppg": "raw.ppg",
    "core_temp": "raw.core_temp",
    "ambient_temp": "raw.ambient_temp",
    "impact": "raw.impact",
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
    """Builds an INSERT INTO … SQL statement for a single packet."""
    ptype = pkt.get("type")
    if ptype not in TABLES:
        return None

    table = TABLES[ptype]

    # Timestamp
    ts = pkt.get("ts") or datetime.utcnow().isoformat()

    # PACKET-SPECIFIC MAPPINGS ----------------------
    if ptype == "eeg_raw":
        cols = ["ts", "session_id", "device_id", "channel", "samples"]
        vals = [
            pg_literal(ts),
            pg_literal(session_id),
            pg_literal(device_id),
            pg_literal(pkt.get("channel")),
            pg_literal(pkt.get("samples")),
        ]

    elif ptype == "eeg_bands":
        cols = ["ts", "session_id", "device_id", "delta", "theta", "alpha", "beta", "gamma"]
        vals = [
            pg_literal(ts), pg_literal(session_id), pg_literal(device_id),
            pg_literal(pkt.get("delta")), pg_literal(pkt.get("theta")),
            pg_literal(pkt.get("alpha")), pg_literal(pkt.get("beta")), pg_literal(pkt.get("gamma"))
        ]

    elif ptype == "clinical_metrics":
        cols = ["ts", "session_id", "device_id", "alertness", "cognitive_load", "fatigue", "zone_durations"]
        vals = [
            pg_literal(ts), pg_literal(session_id), pg_literal(device_id),
            pg_literal(pkt.get("alertness")),
            pg_literal(pkt.get("cognitive_load")),
            pg_literal(pkt.get("fatigue")),
            pg_literal(pkt.get("zone_durations")),
        ]

    elif ptype == "hydration":
        cols = ["ts", "session_id", "device_id", "value", "zone", "zone_durations"]
        vals = [
            pg_literal(ts), pg_literal(session_id), pg_literal(device_id),
            pg_literal(pkt.get("value")),
            pg_literal(pkt.get("zone")),
            pg_literal(pkt.get("zone_durations")),
        ]

    elif ptype == "ppg":
        cols = ["ts", "session_id", "device_id", "heart_rate", "spo2", "hr_zone_durations", "spo2_zone_durations"]
        vals = [
            pg_literal(ts), pg_literal(session_id), pg_literal(device_id),
            pg_literal(pkt.get("heart_rate")),
            pg_literal(pkt.get("spo2")),
            pg_literal(pkt.get("hr_zone_durations")),
            pg_literal(pkt.get("spo2_zone_durations")),
        ]

    elif ptype == "core_temp":
        cols = ["ts", "session_id", "device_id", "temperature_c", "zone_durations"]
        vals = [
            pg_literal(ts), pg_literal(session_id), pg_literal(device_id),
            pg_literal(pkt.get("temperature_c")),
            pg_literal(pkt.get("zone_durations")),
        ]

    elif ptype == "ambient_temp":
        cols = ["ts", "session_id", "device_id", "temperature_c"]
        vals = [
            pg_literal(ts), pg_literal(session_id), pg_literal(device_id),
            pg_literal(pkt.get("temperature_c"))
        ]

    elif ptype == "impact":
        cols = ["ts", "session_id", "device_id", "current_magnitude_g", "zone_counts"]
        vals = [
            pg_literal(ts), pg_literal(session_id), pg_literal(device_id),
            pg_literal(pkt.get("current_magnitude_g")),
            pg_literal(pkt.get("zone_counts"))
        ]

    else:
        return None

    col_sql = ", ".join(cols)
    val_sql = ", ".join(vals)
    return f"INSERT INTO {table} ({col_sql}) VALUES ({val_sql});"


# ----------------------------------------------
# MAIN
# ----------------------------------------------

def main():
    print("Connecting to Arduino...")
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)

    session_id, device_id = new_ids()

    outfile_name = f"session_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.sql"
    outfile_path = OUTPUT_DIR / outfile_name

    with outfile_path.open("w", encoding="utf-8") as f:
        f.write("-- BeAST End-of-Session Export\n")
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

                # Expect JSON
                try:
                    msg = json.loads(line)
                except:
                    continue

                if msg.get("type") == "data_batch":
                    for pkt in msg.get("batch", []):
                        insert_sql = build_insert(pkt, session_id, device_id)
                        if insert_sql:
                            f.write(insert_sql + "\n")

        except KeyboardInterrupt:
            print("Session ended by user.\n")

        f.write("\nCOMMIT;\n")

    print(f"Session SQL saved: {outfile_path}")


if __name__ == "__main__":
    main()
