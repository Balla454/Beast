#!/bin/bash
# BeAST Unified Installer
# Installs everything for a new or used device.
# Usage: ./install.sh

set -e  # Exit on error

# -----------------------------------------------------------------------------
# 1. Pre-flight Checks
# -----------------------------------------------------------------------------

# Check if running as root - don't do that!
if [ "$EUID" -eq 0 ]; then
    echo "ERROR: Do not run this script as root or with sudo!"
    echo "Run it as your normal user: ./install.sh"
    exit 1
fi

echo "==================================="
echo "BeAST Unified Installer"
echo "==================================="

BEAST_USER=$(whoami)
BEAST_HOME=$(eval echo ~$BEAST_USER)
BEAST_DIR="$BEAST_HOME/Beast/BeAST_Pi_Code"
VENV_DIR="$BEAST_HOME/.beast-venv"

echo "User: $BEAST_USER"
echo "Home: $BEAST_HOME"
echo "Dir:  $BEAST_DIR"
echo ""

# -----------------------------------------------------------------------------
# 2. System Dependencies
# -----------------------------------------------------------------------------
echo "[1/6] Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y \
    python3-full \
    python3-pip \
    python3-venv \
    portaudio19-dev \
    libasound2-dev \
    libsndfile1-dev \
    ffmpeg \
    espeak \
    espeak-ng \
    libopenblas-dev \
    git

# -----------------------------------------------------------------------------
# 2.5. Ollama Setup
# -----------------------------------------------------------------------------
echo "[1.5/6] Installing Ollama..."
if ! command -v ollama &> /dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "  - Ollama already installed."
fi

echo "  - Pulling gemma2:2b model..."
# Ensure ollama is running before pulling
if ! pgrep -x "ollama" > /dev/null; then
    echo "  - Starting Ollama server..."
    ollama serve > /dev/null 2>&1 &
    sleep 5
fi
ollama pull gemma2:2b

# -----------------------------------------------------------------------------
# 3. Virtual Environment Setup (Clean Install)
# -----------------------------------------------------------------------------
echo "[2/6] Setting up Python virtual environment..."

# Remove existing venv to ensure a clean slate (handles "used device" case)
if [ -d "$VENV_DIR" ]; then
    echo "  - Removing old virtual environment..."
    rm -rf "$VENV_DIR"
fi

# Create new venv
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# Upgrade pip
pip install --upgrade pip -q

# -----------------------------------------------------------------------------
# 4. Python Dependencies
# -----------------------------------------------------------------------------
echo "[3/6] Installing Python libraries..."

# Voice Assistant Requirements
if [ -f "$BEAST_DIR/TheBeast/TheBeast/beast/requirements.txt" ]; then
    echo "  - Installing Voice Assistant dependencies..."
    pip install -r "$BEAST_DIR/TheBeast/TheBeast/beast/requirements.txt"
else
    echo "ERROR: Voice Assistant requirements.txt not found!"
    exit 1
fi

# Data Collector Requirements
echo "  - Installing Data Collector dependencies..."
pip install pyserial psycopg2-binary

# -----------------------------------------------------------------------------
# 5. Model Downloads (Offline Mode)
# -----------------------------------------------------------------------------
echo "[4/6] Downloading AI models..."
# Use the existing install.py to handle model downloads
# We are already in the venv, so we run python directly
python "$BEAST_DIR/TheBeast/TheBeast/beast/install.py" --models

# -----------------------------------------------------------------------------
# 6. System Configuration
# -----------------------------------------------------------------------------
echo "[5/6] Configuring system..."

# Add user to audio group
sudo usermod -aG audio "$BEAST_USER"

# Fix file ownership
sudo chown -R "$BEAST_USER:$BEAST_USER" "$BEAST_HOME/Beast"

# Create environment file for services
sudo mkdir -p /etc/beast
echo "BEAST_USER=$BEAST_USER" | sudo tee /etc/beast/environment > /dev/null
echo "BEAST_HOME=$BEAST_HOME" | sudo tee -a /etc/beast/environment > /dev/null

# -----------------------------------------------------------------------------
# 7. Service Installation
# -----------------------------------------------------------------------------
echo "[6/6] Installing services..."

# Stop old services
sudo systemctl stop beast-voice.service 2>/dev/null || true
sudo systemctl stop beast-synthetic.service 2>/dev/null || true
sudo systemctl stop beast-arduino.service 2>/dev/null || true

# Generate service files with correct paths/user
echo "  - Generating beast-arduino.service..."
cat <<EOF | sudo tee /etc/systemd/system/beast-arduino.service > /dev/null
[Unit]
Description=BeAST Arduino Data Collector
After=network.target postgresql.service
Wants=postgresql.service
# Wait for Arduino to be recognized
After=dev-ttyACM0.device

[Service]
Type=simple
User=$BEAST_USER
WorkingDirectory=$BEAST_DIR/Live Connections Simulator Test
Environment="PATH=$VENV_DIR/bin:/usr/local/bin:/usr/bin:/bin"
Environment="BEAST_DB_PASSWORD=beast"
ExecStart=$VENV_DIR/bin/python beast_arduino_to_sql_updated.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

echo "  - Generating beast-voice.service..."
cat <<EOF | sudo tee /etc/systemd/system/beast-voice.service > /dev/null
[Unit]
Description=BeAST Voice Assistant
After=network.target sound.target postgresql.service
Wants=postgresql.service

[Service]
Type=simple
EnvironmentFile=/etc/beast/environment
User=$BEAST_USER
WorkingDirectory=$BEAST_DIR/TheBeast/TheBeast/beast
Environment="PATH=$VENV_DIR/bin:/usr/local/bin:/usr/bin:/bin"
Environment="HF_HUB_OFFLINE=1"
Environment="TRANSFORMERS_OFFLINE=1"
Environment="BEAST_DB_PASSWORD=beast"
ExecStart=$VENV_DIR/bin/python main.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

echo "  - Generating beast-synthetic.service..."
cat <<EOF | sudo tee /etc/systemd/system/beast-synthetic.service > /dev/null
[Unit]
Description=BeAST Synthetic Data Service
After=network.target

[Service]
Type=simple
User=$BEAST_USER
WorkingDirectory=$BEAST_DIR/Live Connections Simulator Test
Environment="PATH=$VENV_DIR/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=$VENV_DIR/bin/python beast_synthetic_playback.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Reload and enable
sudo systemctl daemon-reload
sudo systemctl enable beast-voice.service
sudo systemctl enable beast-synthetic.service
sudo systemctl enable beast-arduino.service

echo "Starting services..."
sudo systemctl start beast-voice.service
sudo systemctl start beast-arduino.service

echo ""
echo "==================================="
echo "Installation Complete!"
echo "==================================="
echo "Services have been started."
echo "To check status:"
echo "  systemctl status beast-voice"
echo ""
