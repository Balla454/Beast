# Copilot Instructions for BeAST

This repository contains the BeAST (Biometric Enhancement & Adaptive Sensing Technology) system.

## Project Architecture

The system is divided into two main functional areas within `BeAST_Pi_Code/`:

1.  **Data Collection & Management** (`Live Connections Simulator Test/`, `BeAST Schema/`)
    -   **Ingestion:** `beast_arduino_to_sql.py` reads serial data from Arduino sensors.
    -   **Storage:** Data is exported to SQL session files in `beast_session_exports/` (PostgreSQL dialect).
    -   **Schema:** Defined in `BeAST Schema/beast_schema.sql`.
    -   **Entry Point:** `beast_session_runner.sh` handles environment setup and execution.

2.  **Voice Assistant & RAG** (`TheBeast/TheBeast/`)
    -   **Core Loop:** `beast/main.py` manages the wake word -> STT -> RAG -> TTS loop.
    -   **Offline First:** The system runs fully offline (`HF_HUB_OFFLINE=1`). Models must be pre-downloaded.
    -   **RAG System:** `rag_system.py` and `enhanced_rag_chat.py` handle knowledge retrieval.
    -   **Sensor Fusion:** `beast_sensordata/` processes physiological metrics.

## Critical Workflows

-   **Running Data Collection:**
    Use the session runner which handles venv and user detection:
    ```bash
    ./beast_session_runner.sh
    ```

-   **Running Voice Assistant:**
    Navigate to `TheBeast/TheBeast/beast/` and run:
    ```bash
    python3 main.py
    ```

-   **Service Management:**
    Systemd services (`beast-arduino.service`, `beast-voice.service`) use specifiers (`%u`, `%h`) for portability.
    -   **Do not hardcode paths** in service files; use these specifiers.

## Coding Conventions

-   **Path Handling:**
    -   Use `pathlib.Path` for all file operations.
    -   **Never hardcode absolute paths** (e.g., `/home/pi`). Use relative paths or environment detection as seen in `beast_session_runner.sh`.
    -   Respect `HF_HUB_OFFLINE` environment variable for ML components.

-   **Database:**
    -   Primary data format is PostgreSQL-compatible SQL.
    -   When modifying `beast_arduino_to_sql.py`, ensure `TABLES` mapping matches `beast_schema.sql`.
    -   `LocalDatabaseManager._initialize_database()` in `beast_realtime_processor.py` defines the local SQLite schema for real-time metrics.

-   **RAG & ML:**
    -   **Graceful Degradation:** Check for model availability (`*_AVAILABLE` flags) before execution.
    -   **Caching:** `DatasetKnowledgeBase` caches embeddings. Preserve `_save_to_cache` logic.
    -   **Feature Naming:** Maintain consistency in `BeASTSensorFusion` feature names (e.g., `left_hr_mean`) to avoid breaking downstream analysis.

## Key Files & Components

-   **Entry Points:**
    -   `BeAST_Pi_Code/beast_session_runner.sh`: Data collection.
    -   `BeAST_Pi_Code/TheBeast/TheBeast/beast/main.py`: Voice assistant.
    -   `BeAST_Pi_Code/TheBeast/TheBeast/enhanced_rag_chat.py`: Interactive RAG chat.

-   **Data Processing:**
    -   `BeAST_Pi_Code/Live Connections Simulator Test/beast_arduino_to_sql.py`: Serial data parser.
    -   `BeAST_Pi_Code/TheBeast/TheBeast/beast_sensordata/sensor_fusion.py`: Sensor alignment and feature extraction.
    -   `BeAST_Pi_Code/TheBeast/TheBeast/Data Analysis/BeAST Processor Code/beast_realtime_processor.py`: Real-time metrics pipeline.

## AI Agent Guidelines

-   **Context Awareness:** Determine if you are working on the **Data Collector** (Arduino/SQL) or the **Voice Assistant** (Python/RAG).
-   **Offline Constraint:** Assume no internet access for the Voice Assistant. Do not suggest `pip install` or `model.download()` inside runtime code.
-   **Systemd:** When editing services, preserve the `%u` and `%h` specifiers.
-   **Modifying RAG:** When changing `RAGSystem`, ensure `install_dependencies()` and `README.md` are updated if dependencies change.
