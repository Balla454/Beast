#!/bin/bash
# BeAST Service Setup Script
# Automatically configures and installs beast services for any user

set -e  # Exit on error

echo "==================================="
echo "BeAST Service Setup"
echo "==================================="

# Get current user and home directory
BEAST_USER=$(whoami)
BEAST_HOME=$(eval echo ~$BEAST_USER)
BEAST_DIR="$BEAST_HOME/Beast/BeAST_Pi_Code"

echo "User: $BEAST_USER"
echo "Home: $BEAST_HOME"
echo "Beast Directory: $BEAST_DIR"
echo ""

# Check if Beast directory exists
if [ ! -d "$BEAST_DIR" ]; then
    echo "Error: Beast directory not found at $BEAST_DIR"
    exit 1
fi

# 1. Create Python virtual environment
echo "[1/7] Creating Python virtual environment..."
if [ -d "$BEAST_HOME/.beast-venv" ]; then
    echo "  - Removing existing virtual environment (not portable)..."
    rm -rf "$BEAST_HOME/.beast-venv"
fi
python3 -m venv "$BEAST_HOME/.beast-venv"

# 2. Install requirements
echo "[2/7] Installing Python dependencies..."
source "$BEAST_HOME/.beast-venv/bin/activate"
pip install --upgrade pip -q
pip install -r "$BEAST_DIR/TheBeast/TheBeast/beast/requirements.txt" -q
deactivate

# 3. Add user to audio group
echo "[3/7] Adding user to audio group..."
sudo usermod -aG audio "$BEAST_USER"

# 4. Fix file ownership
echo "[4/7] Setting file ownership..."
sudo chown -R "$BEAST_USER:$BEAST_USER" "$BEAST_HOME/Beast"

# 5. Symlink service files
echo "[5/7] Linking service files..."

sudo ln -sf "$BEAST_DIR/beast-voice.service" /etc/systemd/system/beast-voice.service
sudo ln -sf "$BEAST_DIR/beast-synthetic.service" /etc/systemd/system/beast-synthetic.service

# 6. Reload systemd and enable services
echo "[6/7] Enabling services..."
sudo systemctl daemon-reload
sudo systemctl enable beast-voice.service
sudo systemctl enable beast-synthetic.service

# 7. Start services
echo "[7/7] Starting services..."
sudo systemctl start beast-voice.service
sudo systemctl start beast-synthetic.service

echo ""
echo "==================================="
echo "Setup Complete!"
echo "==================================="
echo ""
echo "Services installed and started:"
echo "  - beast-voice.service (Voice Assistant)"
echo "  - beast-synthetic.service (Synthetic Data)"
echo ""
echo "Check status with:"
echo "  systemctl status beast-voice.service"
echo "  systemctl status beast-synthetic.service"
echo ""
echo "View logs with:"
echo "  journalctl -u beast-voice.service -f"
echo "  journalctl -u beast-synthetic.service -f"
echo ""
