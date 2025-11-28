# Copilot instructions for TheBeast repository

This file gives practical, repo-specific guidance for AI coding agents (Copilot-like) to be immediately productive.

Keep guidance concise and actionable; reference files and examples in the repository.

Overview
- **Big picture:** This repo combines an interactive RAG-powered chat interface (`enhanced_rag_chat.py` + `rag_system.py`) with a set of BeAST sensor data processing tools (`beast_sensordata/`, `Data Analysis/BeAST Processor Code/`). The RAG system indexes organized research datasets and uses local LLMs. The realtime processor computes 10 physiological/cognitive metrics and writes to a local SQLite DB.

Key components & boundaries
- **Interactive chat / RAG:** `enhanced_rag_chat.py` (entrypoint) uses `RAGSystem` from `rag_system.py`. RAG responsibilities: dataset ingestion, embeddings (SentenceTransformers), FAISS indexing, and generation via a local model path (Gemma/Gemma3n). See `RAGSystem`, `DatasetKnowledgeBase` in `rag_system.py` for indexing, caching, and model-loading patterns.
- **Sensor fusion & feature extraction:** `beast_sensordata/sensor_fusion.py` contains `BeASTSensorFusion` for bilateral sensor alignment, PPG/EEG/IMU/temp/bioz feature extraction and a consistent feature-naming scheme (e.g., `left_hr_mean`, `eeg_ch1_bilateral_diff`). Use these feature names verbatim when integrating ML models or tests.
- **Real-time processing & metrics pipeline:** `Data Analysis/BeAST Processor Code/beast_realtime_processor.py` is the production-style real-time pipeline. It defines `SignalPreprocessor`, `FeatureExtractor`, `MetricCalculator`, `ZoneClassifier`, and `LocalDatabaseManager`. Metric scaling, zone thresholds, and DB schema are explicitly encoded — changes here are critical and must be cross-checked across the repo.

Developer workflows (how to run and test)
- **Run interactive chat (local dev):** `python3 enhanced_rag_chat.py`. The script detects optional components (TTS, STT) and logs availability. If local models or dataset paths are missing, the script falls back gracefully.
- **Run RAG demo / test:** `python3 rag_system.py` runs a local demo that attempts to load embedding libraries and prompts. If embeddings or FAISS are missing, `rag_system.install_dependencies()` exists to pip-install core packages.
- **Run realtime processor (test):** In `Data Analysis/BeAST Processor Code`, run `python3 beast_realtime_processor.py` — it will generate synthetic JSONL and exercise the pipeline, producing `beast_local.db` in the working directory.
- **Unit tests:** There is a `test_sensor_fusion.py` near `beast_sensordata/` to validate sensor parsing/feature extraction. Run `python3 -m pytest test_sensor_fusion.py` from `beast_sensordata/` or `pytest` at repo root if `pytest` is installed.

Conventions and patterns to follow (repo-specific)
- **Graceful optional dependencies:** Many modules attempt imports and set `*_AVAILABLE` flags (e.g., `RAG_AVAILABLE`, `TTS_AVAILABLE`). Follow this pattern: prefer non-fatal fallbacks and informative logging rather than hard failures.
- **Paths are often absolute in examples:** Look for dataset/model paths like `/Users/collinball/Applications/TheBeast/dataset` and `organized/models/gemma3n`. For changes, prefer configurable env vars or parameters rather than hardcoding new absolute paths.
- **Indexing & caching:** `DatasetKnowledgeBase` caches embeddings and FAISS index under a `cache` sibling to the dataset root. When modifying ingestion logic, preserve or update `_save_to_cache` / `_load_from_cache` formats and metadata files (`metadata.json`, `faiss_index.bin`) to avoid invalidating caches.
- **Feature naming & size expectations:** `BeASTSensorFusion.create_feature_vector` and `BeASTRealtimeProcessor` expect specific feature dictionaries and vector orders. Keep feature-name stability to avoid breaking downstream DB or ML consumers.
- **DB schema is authoritative:** `LocalDatabaseManager._initialize_database()` is the canonical schema. If you change metric names or types, update the schema and migration guidance together.

Integration points & external dependencies
- **Embedding & index:** `sentence_transformers` + `faiss` (faiss-cpu) for embeddings and search. `DatasetKnowledgeBase` uses `all-MiniLM-L6-v2` by default; any change must handle batching and normalized embeddings.
- **LLM:** `transformers` + `torch` for local Gemma model loading. `RAGSystem.load_language_model()` expects `model_path` containing local files and uses `trust_remote_code=True` and `local_files_only=True`.
- **TTS/STT:** Optional external modules: `piper-tts` (or system TTS) and `moonshine_stt`/`faster-whisper-tiny`. These are optional and guarded with availability checks.

How to modify code safely (advice for AI agents)
- When changing any model-loading or dependency logic, update both `install_dependencies()` in `rag_system.py` and the README notes in `README.md`.
- If you modify prompt/context assembly in `RAGSystem._create_augmented_prompt`, keep the `max_context_chars` and sampling parameters in `_generate_response` in sync — those were tuned for CPU/speed tradeoffs.
- For changes that affect DB fields or metric names, add a small migration helper in `beast_realtime_processor.py` adjacent to `LocalDatabaseManager._initialize_database()` and update any code that writes to those columns.

Examples from the codebase (use these snippets when reasoning)
- RAG call site: `EnhancedGemmaRAGChat.process_query()` in `enhanced_rag_chat.py` — use `rag_system.generate_rag_response(user_input)` and honor `show_sources` flag when appending sources.
- Feature vector creation: `BeASTSensorFusion.create_feature_vector(data)` returns numpy vector and keeps `feature_names` on the instance — tests and ML code rely on ordering.
- Realtime test harness: `beast_realtime_processor.generate_synthetic_jsonl()` used by `main()` for quick end-to-end verification.

What not to change without human approval
- The DB schema and metric-to-zone thresholds in `beast_realtime_processor.py` (risk of breaking downstream analytics).
- Default prompt and truncation heuristics in `rag_system.py` (they are tuned for local CPU constraints).

Where to look first for common tasks
- Add a new RAG data parser: update `DatasetKnowledgeBase._process_*` methods and the cache save/load logic.
- Add new sensor feature: update `beast_sensordata/sensor_fusion.py` (`feature_names`) and `Data Analysis/BeAST Processor Code/FeatureExtractor` if realtime pipeline needs it.
- Add tests: mirror the style in `beast_sensordata/test_sensor_fusion.py` and include small synthetic JSON/CSV fixtures.

Questions for the maintainer (ask these if ambiguous)
- Which dataset/model paths should be treated as configurable (env var) vs locked examples? Provide preferred env var names.
- Are there intended consumers of the SQLite schema beyond the repo (e.g., downstream dashboards) that require backward compatibility?

End of file. After you make changes, run the local demos: `python3 rag_system.py` and `python3 Data\\ Analysis/BeAST\\ Processor\\ Code/beast_realtime_processor.py` to smoke-test.
