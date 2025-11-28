#!/usr/bin/env python3
import asyncio
import json
import logging
import random
from datetime import datetime, timezone

import websockets

# ---------------------------------------------------
# Config
# ---------------------------------------------------
WS_HOST = "0.0.0.0"   # allow dashboard from other machines; use "localhost" if only local
WS_PORT = 8765

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# ---------------------------------------------------
# Helpers: synthetic packet generators
# ---------------------------------------------------
def iso_now():
    return datetime.now(timezone.utc).isoformat()

def make_eeg_raw_packet():
    return {
        "type": "eeg_raw",
        "ts": iso_now(),
        "channel": random.randint(1, 4),  # 4 channels example
        "samples": [random.randint(-100, 100) for _ in range(10)],
    }

def make_eeg_bands_packet():
    return {
        "type": "eeg_bands",
        "ts": iso_now(),
        "delta": round(random.uniform(0.5, 3.0), 3),
        "theta": round(random.uniform(1.0, 4.0), 3),
        "alpha": round(random.uniform(2.0, 6.0), 3),
        "beta":  round(random.uniform(3.0, 8.0), 3),
        "gamma": round(random.uniform(5.0, 10.0), 3),
    }

def make_zone_durations():
    # simple synthetic cumulative zone durations
    return {
        "GREEN": random.randint(60, 600),
        "YELLOW": random.randint(0, 300),
        "ORANGE": random.randint(0, 120),
        "RED": random.randint(0, 60),
    }

def make_clinical_metrics_packet():
    return {
        "type": "clinical_metrics",
        "ts": iso_now(),
        "alertness": random.randint(40, 95),
        "cognitive_load": random.randint(20, 90),
        "fatigue": random.randint(10, 80),
        "zone_durations": make_zone_durations(),
    }

def make_hydration_packet():
    return {
        "type": "hydration",
        "ts": iso_now(),
        "value": round(random.uniform(40.0, 60.0), 2),
        "zone": random.choice(["GREEN", "YELLOW", "ORANGE", "RED"]),
        "zone_durations": make_zone_durations(),
    }

def make_ppg_packet():
    return {
        "type": "ppg",
        "ts": iso_now(),
        "heart_rate": random.randint(60, 110),
        "spo2": random.randint(92, 99),
        "hr_zone_durations": make_zone_durations(),
        "spo2_zone_durations": make_zone_durations(),
    }

def make_core_temp_packet():
    return {
        "type": "core_temp",
        "ts": iso_now(),
        "temperature_c": round(random.uniform(36.2, 37.8), 2),
        "zone_durations": make_zone_durations(),
    }

def make_ambient_temp_packet():
    return {
        "type": "ambient_temp",
        "ts": iso_now(),
        "temperature_c": round(random.uniform(20.0, 30.0), 2),
    }

def make_impact_packet():
    return {
        "type": "impact",
        "ts": iso_now(),
        "current_magnitude_g": round(random.uniform(0.0, 5.0), 2),
        "zone_counts": {
            "GREEN": random.randint(0, 10),
            "YELLOW": random.randint(0, 5),
            "ORANGE": random.randint(0, 3),
            "RED": random.randint(0, 1),
        },
    }

# ---------------------------------------------------
# Connection handler
# ---------------------------------------------------
async def handle_client(websocket):
    logging.info("✅ Dashboard connected: %s", websocket.remote_address)
    current_mode = None
    session_active = False
    session_id = "sim-session-001"
    device_id = "sim-device-001"

    # Send a connected message
    await websocket.send(json.dumps({
        "type": "connected",
        "message": "Connected to BeAST simulated live data stream",
        "timestamp": iso_now()
    }))

    async def streaming_loop():
        nonlocal current_mode, session_active
        while session_active and current_mode is not None:
            batch = []

            if current_mode == "L1":
                # L1: eeg_raw (12 packets/sec at spec; we fake as 1 *batch* per second)
                for _ in range(12):
                    batch.append(make_eeg_raw_packet())

            elif current_mode == "L2":
                # L2: eeg_bands (1 packet/sec)
                batch.append(make_eeg_bands_packet())

            elif current_mode == "L3":
                # L3: multiple clinical/physio metrics (1 Hz each)
                batch.append(make_clinical_metrics_packet())
                batch.append(make_hydration_packet())
                batch.append(make_ppg_packet())
                batch.append(make_core_temp_packet())
                batch.append(make_ambient_temp_packet())
                batch.append(make_impact_packet())

            if batch:
                msg = {
                    "type": "data_batch",
                    "batch": batch,
                    "session_id": session_id,
                    "device_id": device_id,
                    "timestamp": iso_now(),
                }
                await websocket.send(json.dumps(msg))

            await asyncio.sleep(1.0)  # 1 Hz

    stream_task = None

    try:
        async for raw_msg in websocket:
            # Accept either plain "L1"/"L2"/"L3"/"S" or JSON {"command":"L1"}
            try:
                data = json.loads(raw_msg)
                command = data.get("command", "").strip().upper()
            except json.JSONDecodeError:
                command = raw_msg.strip().upper()

            logging.info("📥 Command from dashboard: %s", command)

            if command in ("L1", "L2", "L3"):
                current_mode = command
                session_active = True

                # Cancel old stream if any
                if stream_task is not None and not stream_task.done():
                    stream_task.cancel()

                await websocket.send(json.dumps({
                    "status": "live_started",
                    "mode": current_mode,
                    "session_id": session_id,
                    "device_id": device_id,
                    "timestamp": iso_now()
                }))

                stream_task = asyncio.create_task(streaming_loop())

            elif command == "S":
                session_active = False
                current_mode = None
                if stream_task is not None and not stream_task.done():
                    stream_task.cancel()

                await websocket.send(json.dumps({
                    "status": "live_stopped",
                    "session_id": session_id,
                    "device_id": device_id,
                    "timestamp": iso_now()
                }))
            else:
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": f"Unknown command: {command}"
                }))

    except websockets.exceptions.ConnectionClosed:
        logging.info("🔌 Dashboard disconnected: %s", websocket.remote_address)
        if stream_task is not None and not stream_task.done():
            stream_task.cancel()


async def main():
    logging.info("============================================================")
    logging.info("BeAST Simulated Live Data Stream Service")
    logging.info("No Arduino required - synthetic packets only")
    logging.info("============================================================")
    logging.info("WebSocket: ws://%s:%d", WS_HOST, WS_PORT)

    async with websockets.serve(handle_client, WS_HOST, WS_PORT):
        logging.info("✅ WebSocket server started; waiting for dashboard connections...")
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Shutting down simulator...")
