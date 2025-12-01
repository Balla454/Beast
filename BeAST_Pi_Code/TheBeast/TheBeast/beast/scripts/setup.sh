#!/bin/bash
# =============================================================================
# beast System Setup Script
# =============================================================================
# Run this once on a fresh Raspberry Pi 5 to configure:
# - System dependencies
# - Python environment
# - Power button handler (triggerhappy)
# - SSH keys for sync
# - Systemd service for auto-start
#
# Environment variables (set before running or use defaults):
#   BEAST_HOME       - Installation directory (default: ~/beast)
#   BEAST_DATA_DIR   - Data directory (default: $BEAST_HOME/data)
#   BEAST_MODELS_DIR - Models directory (default: $BEAST_HOME/models)
# =============================================================================

set -e

# Configuration via environment variables
BEAST_HOME="${BEAST_HOME:-$HOME/beast}"
DATA_DIR="${BEAST_DATA_DIR:-$BEAST_HOME/data}"
MODELS_DIR="${BEAST_MODELS_DIR:-$BEAST_HOME/models}"
LOGS_DIR="${BEAST_LOG_DIR:-$BEAST_HOME/logs}"
VENV_DIR="${BEAST_VENV:-$BEAST_HOME/.venv}"
HOST_PC="${BEAST_BACKUP_HOST:-}"

echo "=========================================="
echo "beast System Setup"
echo "=========================================="
echo "BEAST_HOME: $BEAST_HOME"
echo "DATA_DIR: $DATA_DIR"
echo "MODELS_DIR: $MODELS_DIR"
echo "=========================================="

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo "Please run as regular user (not root)"
    exit 1
fi

# Update system packages
echo ""
echo ">>> Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install system dependencies
echo ""
echo ">>> Installing system dependencies..."
sudo apt install -y \
    python3-full \
    python3-pip \
    python3-venv \
    git \
    triggerhappy \
    espeak \
    espeak-ng \
    portaudio19-dev \
    libasound2-dev \
    libsndfile1-dev \
    ffmpeg \
    sqlite3 \
    sshpass \
    libatlas-base-dev \
    libopenblas-dev \
    libjpeg-dev \
    zlib1g-dev

# Create directory structure
echo ""
echo ">>> Creating directory structure..."
mkdir -p "$BEAST_HOME"
mkdir -p "$DATA_DIR"
mkdir -p "$MODELS_DIR/tts"
mkdir -p "$MODELS_DIR/stt"
mkdir -p "$MODELS_DIR/rag"
mkdir -p "$LOGS_DIR"
mkdir -p "$BEAST_HOME/backups"

# Create Python virtual environment
echo ""
echo ">>> Setting up Python virtual environment..."
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# Install Python packages
echo ""
echo ">>> Installing Python packages..."
pip install --upgrade pip wheel setuptools

# Core packages
pip install \
    numpy \
    scipy \
    pyyaml \
    sounddevice \
    soundfile \
    pyaudio

# AI/ML packages (may take a while on Pi)
echo ""
echo ">>> Installing AI/ML packages (this may take a while)..."
pip install \
    torch --index-url https://download.pytorch.org/whl/cpu \
    transformers \
    sentence-transformers \
    faiss-cpu

# Voice packages
echo ""
echo ">>> Installing voice packages..."
pip install \
    faster-whisper \
    piper-tts \
    webrtcvad

# Database
pip install psycopg2-binary

deactivate

# Set up triggerhappy for power button
echo ""
echo ">>> Configuring power button handler..."

# Create triggerhappy config
sudo tee /etc/triggerhappy/triggers.d/beast-power.conf > /dev/null << EOF
# beast power button handler
# Triggers data sync and shutdown on power button press
KEY_POWER    1    $BEAST_HOME/scripts/zip.sh
EOF

# Make scripts executable
chmod +x "$BEAST_HOME/scripts/"*.sh 2>/dev/null || true

# Enable triggerhappy service
sudo systemctl enable triggerhappy
sudo systemctl restart triggerhappy

# Set up SSH keys for passwordless sync
echo ""
echo ">>> Setting up SSH keys for host sync..."

if [ ! -f "$HOME/.ssh/id_rsa" ]; then
    ssh-keygen -t rsa -b 4096 -f "$HOME/.ssh/id_rsa" -N "" -q
    echo "SSH key generated."
    echo ""
    echo "!!! ACTION REQUIRED !!!"
    if [ -n "$HOST_PC" ]; then
        echo "Copy SSH key to host PC with:"
        echo "  ssh-copy-id $(whoami)@${HOST_PC}"
    else
        echo "Set BEAST_BACKUP_HOST and copy SSH key to host PC"
    fi
    echo ""
else
    echo "SSH key already exists."
fi

# Create systemd service for beast
echo ""
echo ">>> Creating systemd service..."

sudo tee /etc/systemd/system/beast.service > /dev/null << EOF
[Unit]
Description=beast Wearable AI System
After=network.target sound.target

[Service]
Type=simple
User=$(whoami)
WorkingDirectory=$BEAST_HOME
Environment="PATH=$VENV_DIR/bin:/usr/local/bin:/usr/bin:/bin"
Environment="BEAST_HOME=$BEAST_HOME"
Environment="BEAST_DATA_DIR=$DATA_DIR"
Environment="BEAST_MODELS_DIR=$MODELS_DIR"
Environment="BEAST_LOG_DIR=$LOGS_DIR"
Environment="BEAST_CONFIG=$BEAST_HOME/config.yaml"
ExecStart=$VENV_DIR/bin/python3 $BEAST_HOME/main.py
Restart=always
RestartSec=10

# Audio access
SupplementaryGroups=audio

# Logging
StandardOutput=append:$LOGS_DIR/beast.log
StandardError=append:$LOGS_DIR/beast_error.log

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd and enable service
sudo systemctl daemon-reload
sudo systemctl enable beast.service

# Add user to audio group
sudo usermod -a -G audio $(whoami)

# Create environment file for easy sourcing
cat > "$BEAST_HOME/.env" << EOF
# beast Environment Variables
# Source this file: source $BEAST_HOME/.env

export BEAST_HOME="$BEAST_HOME"
export BEAST_DATA_DIR="$DATA_DIR"
export BEAST_MODELS_DIR="$MODELS_DIR"
export BEAST_LOG_DIR="$LOGS_DIR"
export BEAST_CONFIG="$BEAST_HOME/config.yaml"
export PATH="$VENV_DIR/bin:\$PATH"

# Optional: Set these for backup sync
# export BEAST_BACKUP_HOST="your-host.local"
# export BEAST_BACKUP_USER="$(whoami)"
# export BEAST_BACKUP_PATH="/home/$(whoami)/beast_backups"

# Optional: Database credentials
# export BEAST_DB_HOST="localhost"
# export BEAST_DB_USER="beast"
# export BEAST_DB_PASSWORD=""
EOF

chmod 600 "$BEAST_HOME/.env"

# Final summary
echo ""
echo "=========================================="
echo "beast Setup Complete!"
echo "=========================================="
echo ""
echo "Directory structure:"
echo "  $BEAST_HOME         - Application code"
echo "  $DATA_DIR           - Sensor data & database"
echo "  $MODELS_DIR         - AI models"
echo "  $LOGS_DIR           - Log files"
echo ""
echo "Environment file created: $BEAST_HOME/.env"
echo "  Source it with: source $BEAST_HOME/.env"
echo ""
echo "Next steps:"
echo "  1. Copy application files to $BEAST_HOME"
echo "  2. Download models: python $BEAST_HOME/install.py --models"
echo "  3. Start Ollama: ollama serve && ollama pull gemma2:2b"
echo "  4. Start service: sudo systemctl start beast"
echo "  5. Check status: sudo systemctl status beast"
echo "  6. View logs: journalctl -u beast -f"
echo ""
echo "Power button will sync data and shutdown."
echo ""
