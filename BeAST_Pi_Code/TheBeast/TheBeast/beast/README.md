# beast Edge Deployment
## Raspberry Pi 5 Wearable AI System

This directory contains the complete edge deployment for beast (Bilateral Ear-worn Assessment and Sensing Technology).

## Environment Variables

beast uses environment variables for all paths, making it fully portable:

| Variable | Default | Description |
|----------|---------|-------------|
| `BEAST_HOME` | Script location | Base directory for beast |
| `BEAST_DATA_DIR` | `$BEAST_HOME/data` | Sensor data and database |
| `BEAST_MODELS_DIR` | `$BEAST_HOME/models` | AI model files |
| `BEAST_LOG_DIR` | `$BEAST_HOME/logs` | Log files |
| `BEAST_CONFIG` | `$BEAST_HOME/config.yaml` | Configuration file |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `BEAST_BACKUP_HOST` | `fedora.local` | Host PC for sync |
| `BEAST_BACKUP_USER` | `pi` | SSH user for sync |

## Directory Structure

```
beast/
├── main.py                 # Main entry point
├── config.yaml            # Configuration file
├── requirements.txt       # Python dependencies
├── beast.service          # Systemd service file
├── install.py             # Dependency installer
│
├── voice/                 # Voice interface modules
│   ├── wake_word.py       # Wake word detection ("Beast")
│   ├── speech_to_text.py  # STT (faster-whisper-tiny.en)
│   └── text_to_speech.py  # TTS (Piper + espeak-ng)
│
├── rag/                   # RAG system
│   └── health_rag.py      # Health-focused RAG with Ollama
│
├── processing/            # Sensor processing
│   ├── sensor_interface.py    # Sensor data acquisition
│   ├── feature_extractor.py   # Feature extraction
│   ├── metric_calculator.py   # 10 cognitive/physio metrics
│   └── database_manager.py    # SQLite storage
│
├── utils/                 # Utilities
│   └── announcer.py       # Audio announcements
│
└── scripts/               # Shell scripts
    ├── setup.sh           # First-time setup
    └── zip.sh             # Backup & sync script
```

## Quick Start

### 1. First-Time Setup

On a fresh Raspberry Pi 5:

```bash
# Clone repository to your preferred location
git clone <repo-url> ~/beast
cd ~/beast

# Run setup script
chmod +x scripts/setup.sh
./scripts/setup.sh

# Source the environment (or add to .bashrc)
source .env
```

### 2. Download Models

Download required AI models:

```bash
# Create models directory
mkdir -p "${BEAST_MODELS_DIR:-./models}/tts"

# TTS model (Piper)
wget -O "${BEAST_MODELS_DIR:-./models}/tts/en_US-lessac-medium.onnx" \
    https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx

# STT model (faster-whisper) - downloaded automatically on first use

# LLM (via Ollama)
ollama pull gemma2:2b
```

### 3. Set Up SSH Key for Sync

```bash
# Configure backup host (optional)
export BEAST_BACKUP_HOST=your-pc.local
export BEAST_BACKUP_USER=youruser

# Copy SSH key
ssh-copy-id ${BEAST_BACKUP_USER}@${BEAST_BACKUP_HOST}
```

### 4. Start the Service

```bash
# Start beast
sudo systemctl start beast

# Enable auto-start on boot
sudo systemctl enable beast

# View logs
journalctl -u beast -f
```

## User Experience Flow

1. **Boot**: Pi powers on → beast service starts → Announces "beast is ready"
2. **Wake Word**: User says "Beast" → System announces "I'm listening"
3. **Question**: User asks health question → STT transcribes → RAG generates response
4. **Answer**: TTS speaks the answer → Returns to wake word listening
5. **Shutdown**: Power button pressed → Data syncs to host PC → System shuts down

## Configuration

Edit `config.yaml` to customize:

```yaml
voice:
  wake_word: "beast"        # Wake word phrase
  stt_engine: "whisper"     # whisper, moonshine, or vosk
  tts_engine: "piper"       # piper, espeak, or transformers

rag:
  model_name: "gemma2:2b"   # Ollama model to use
  temperature: 0.7          # Response creativity

sync:
  host: "${BEAST_BACKUP_HOST}"   # Host PC for data sync
  user: "${BEAST_BACKUP_USER}"   # SSH user
```

## Metrics Calculated

The system continuously calculates 10 metrics from sensor data:

| Metric | Description | Good Range |
|--------|-------------|------------|
| Cognitive Load | Mental workload | 0-40 (🟢) |
| Tiredness | Current tiredness | 0-30 (🟢) |
| Fatigue | Accumulated fatigue | 0-25 (🟢) |
| Attention Focus | Focus level | 80-100 (🟢) |
| Stress Index | Physiological stress | 0-30 (🟢) |
| Neurovascular Coupling | Brain blood flow | 80-100 (🟢) |
| Metabolic Stress | Metabolic strain | 0-30 (🟢) |
| Compensation Load | Compensatory effort | 0-30 (🟢) |
| Fatigue Severity | Overall fatigue | 0-25 (��) |
| Attention Capacity | Available attention | 80-100 (🟢) |

## Data Sync

When the power button is pressed:

1. Creates timestamped backup of `${BEAST_DATA_DIR}`
2. Attempts SCP transfer to `${BEAST_BACKUP_HOST}`
3. Cleans old local backups (keeps last 5)
4. Announces "Sync complete" or "Sync failed"
5. Shuts down the system

## Troubleshooting

### No Audio Output

```bash
# Check audio devices
aplay -l

# Test speaker
speaker-test -t wav -c 2

# Check service logs
journalctl -u beast -n 50
```

### Wake Word Not Detected

```bash
# Check microphone
arecord -l

# Test recording
arecord -d 5 -f cd test.wav
aplay test.wav
```

### Sync Fails

```bash
# Test SSH connection
ssh ${BEAST_BACKUP_USER}@${BEAST_BACKUP_HOST} echo "Connected"

# Check network
ping ${BEAST_BACKUP_HOST}

# Run sync manually
./scripts/zip.sh
```

## Development

### Running Locally (macOS/Linux)

```bash
# Set beast home (optional, auto-detected)
export BEAST_HOME=$(pwd)

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run with simulation mode
python main.py
```

### Running Tests

```bash
# Test individual components
python -m processing.sensor_interface
python -m processing.feature_extractor
python -m processing.metric_calculator
python -m voice.text_to_speech

# Run end-to-end test
python test_e2e.py
```

### Verifying Installation

```bash
# Check all dependencies
python install.py --check
```

## License

Proprietary - All rights reserved.
