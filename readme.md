# BeAST — Biometric Enhancement & Adaptive Sensing Technology

A wearable AI system built on Raspberry Pi 5. Arduino sensors stream real-time EEG and biometric data into PostgreSQL, where a local voice assistant queries it using RAG. Runs headlessly via systemd across multiple Pi units.

---

## System Overview

```
Arduino sensors
      │  serial
      ▼
 beast_arduino_to_sql.py  ──►  PostgreSQL
                                    │
                               RAG + FAISS index
                                    │
                              Voice Assistant
                           (wake word → STT → LLM → TTS)
```

**Data flow:** Arduino sends EEG and biometric readings over serial → Python ingests and writes to Postgres → sensor fusion and feature extraction derive cognitive metrics → voice assistant answers queries about your biometric state using a FAISS-indexed RAG over the session data.

---

## Components

### Arduino / Data Collection
- `BeAST_MultiMode.ino` — multi-mode EEG + biometric sensor firmware
- `beast_arduino_to_sql.py` — serial reader, writes to PostgreSQL in real time
- `beast_session_runner.sh` — session entrypoint: auto-creates venv, installs deps, starts collector

### Voice Assistant (`TheBeast/TheBeast/beast/`)
| Module | Role |
|---|---|
| `voice/wake_word.py` | Wake word detection |
| `voice/speech_to_text.py` | faster-whisper (tiny.en, CPU) |
| `voice/text_to_speech.py` | piper-tts via espeak-ng |
| `rag/health_rag.py` | FAISS + sentence-transformers over session data |
| `processing/sensor_interface.py` | Sensor I/O abstraction |
| `processing/feature_extractor.py` | Raw EEG → feature vectors |
| `processing/metric_calculator.py` | Cognitive/biometric metric derivation |
| `processing/database_manager.py` | Postgres read/write |

### Infrastructure
- **Systemd services** — `beast-arduino.service`, `beast-voice.service`, `beast-autostart.service`
- **Power button integration** — safe shutdown + automatic data backup on press
- **Remote backup** — rsync to a central "big database" machine

---

## Setup

### 1. Database
```bash
psql -U postgres -f BeAST_Pi_Code/BeAST\ Schema/beast_schema.sql
```

### 2. Arduino data collector
```bash
cd BeAST_Pi_Code
./beast_session_runner.sh   # handles venv + deps automatically
```

### 3. Voice assistant
```bash
cd BeAST_Pi_Code/TheBeast/TheBeast/beast
python install.py
```

### 4. Systemd services (optional, for headless autostart)
```bash
./BeAST_Pi_Code/setup_services.sh
```

### 5. Power button backup (optional)
```bash
sudo ./BeAST_Pi_Code/setup_power_button.sh
```

### 6. Remote backup server (optional, run on receiving machine)
```bash
sudo ./BeAST_Pi_Code/setup_big_database_computer.sh
```

---

## Stack

| Layer | Tech |
|---|---|
| Hardware | Raspberry Pi 5, Arduino (EEG + biometric sensors) |
| Data ingestion | Python, pyserial, psycopg2 |
| Database | PostgreSQL |
| STT | faster-whisper (tiny.en, CPU) |
| TTS | piper-tts + espeak-ng |
| RAG | FAISS, sentence-transformers, PyTorch |
| Services | systemd |

---

## Multi-Unit Support

Designed to run on a fleet (`beast1`–`beast4`). All scripts auto-detect the current user and use relative paths — no hardcoded usernames or paths anywhere. Systemd service files use `%u` / `%h` specifiers for portability.

---

## Dev / Simulation

No hardware? Use the synthetic data playback:
```bash
python BeAST_Pi_Code/Live\ Connections\ Simulator\ Test/beast_synthetic_playback.py
```

See `Live Connections Simulator Test/DATA_FORMATS.md` for sensor data schemas and `DEVELOPER_SETUP.md` for full dev environment setup.
