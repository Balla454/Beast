#!/bin/bash
# BeAST Service Setup Script
# Automatically configures and installs beast services for any user

set -e  # Exit on error

# Check if running as root - don't do that!
if [ "$EUID" -eq 0 ]; then
    echo "ERROR: Do not run this script as root or with sudo!"
    echo "Run it as your normal user: ./setup_services.sh"
    echo "The script will use sudo internally when needed."
    exit 1
fi

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

# Clean up any accidentally created root venv
if [ -d "/root/.beast-venv" ]; then
    echo "  - Removing incorrect root venv (was run with sudo)..."
    sudo rm -rf "/root/.beast-venv"
fi

python3 -m venv "$BEAST_HOME/.beast-venv"

# 2. Install requirements
echo "[2/7] Installing Python dependencies..."
source "$BEAST_HOME/.beast-venv/bin/activate"
pip install --upgrade pip -q
pip install -r "$BEAST_DIR/TheBeast/TheBeast/beast/requirements.txt"
deactivate

# 3. Add user to audio group
echo "[3/7] Adding user to audio group..."
sudo usermod -aG audio "$BEAST_USER"

# 4. Fix file ownership
echo "[4/8] Setting file ownership..."
sudo chown -R "$BEAST_USER:$BEAST_USER" "$BEAST_HOME/Beast"

# 5. Create environment file for services
echo "[5/8] Creating environment file..."
sudo mkdir -p /etc/beast
echo "BEAST_USER=$BEAST_USER" | sudo tee /etc/beast/environment > /dev/null
echo "BEAST_HOME=$BEAST_HOME" | sudo tee -a /etc/beast/environment > /dev/null

# 6. Symlink service files
echo "[6/8] Linking service files..."

# Stop services if they're running (prevents path conflicts)
sudo systemctl stop beast-voice@$BEAST_USER.service 2>/dev/null || true
sudo systemctl stop beast-voice.service 2>/dev/null || true
sudo systemctl stop beast-synthetic@$BEAST_USER.service 2>/dev/null || true
sudo systemctl stop beast-synthetic.service 2>/dev/null || true

# Remove old service files if they exist (could be copies instead of symlinks)
sudo rm -f /etc/systemd/system/beast-voice.service
sudo rm -f /etc/systemd/system/beast-voice@.service
sudo rm -f /etc/systemd/system/beast-synthetic.service
sudo rm -f /etc/systemd/system/beast-synthetic@.service

# Create symlinks for template services
sudo ln -sf "$BEAST_DIR/beast-voice.service" /etc/systemd/system/beast-voice.service
sudo ln -sf "$BEAST_DIR/beast-synthetic.service" /etc/systemd/system/beast-synthetic.service

# 7. Reload systemd and enable services
echo "[7/8] Enabling services..."
sudo systemctl daemon-reload
sudo systemctl enable beast-voice.service
sudo systemctl enable beast-synthetic.service

# 8. Start services
echo "[8/8] Starting services..."
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
