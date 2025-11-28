#!/bin/bash

# =============================================================================
# BeAST Big Database Computer Setup Script
# =============================================================================
# Run this on the Fedora/Linux machine that will receive backups from the Pi
# =============================================================================

set -e

echo "=========================================="
echo "BeAST Big Database Computer Setup"
echo "=========================================="

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Please run as root: sudo ./setup_big_database_computer.sh"
    exit 1
fi

# -----------------------------------------------------------------------------
# Step 1: Install avahi-daemon for .local hostname resolution
# -----------------------------------------------------------------------------
echo ""
echo "[1/3] Installing avahi-daemon..."

# Detect package manager
if command -v apt &> /dev/null; then
    apt update
    apt install -y avahi-daemon
elif command -v dnf &> /dev/null; then
    dnf install -y avahi
elif command -v yum &> /dev/null; then
    yum install -y avahi
else
    echo "ERROR: Could not detect package manager. Please install avahi manually."
    exit 1
fi

# -----------------------------------------------------------------------------
# Step 2: Enable and start avahi-daemon
# -----------------------------------------------------------------------------
echo ""
echo "[2/3] Enabling avahi-daemon service..."
systemctl enable avahi-daemon
systemctl start avahi-daemon

# -----------------------------------------------------------------------------
# Step 3: Create backups directory
# -----------------------------------------------------------------------------
echo ""
echo "[3/3] Creating backups directory..."

# Get the actual username (not root)
ACTUAL_USER="${SUDO_USER:-$USER}"
BACKUP_DIR="/home/$ACTUAL_USER/backups"

mkdir -p "$BACKUP_DIR"
chown "$ACTUAL_USER:$ACTUAL_USER" "$BACKUP_DIR"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "This computer is now discoverable as: $(hostname).local"
echo "Backup directory created at: $BACKUP_DIR"
echo ""
echo "Make sure the Pi can connect via SSH:"
echo "  1. On the Pi, run: ssh-copy-id $ACTUAL_USER@$(hostname).local"
echo "  2. Test connection: ssh $ACTUAL_USER@$(hostname).local"
echo ""
