#!/bin/bash

# =============================================================================
# BeAST Raspberry Pi Power Button Setup Script
# =============================================================================
# This script configures the power button to trigger a backup before shutdown
# =============================================================================

set -e

echo "=========================================="
echo "BeAST Power Button Setup Script"
echo "=========================================="

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Please run as root: sudo ./setup_power_button.sh"
    exit 1
fi

# -----------------------------------------------------------------------------
# Step 1: Install required packages
# -----------------------------------------------------------------------------
echo ""
echo "[1/6] Installing required packages..."
apt update
apt install -y triggerhappy evtest zip

# -----------------------------------------------------------------------------
# Step 2: Create scripts directory
# -----------------------------------------------------------------------------
echo ""
echo "[2/6] Creating scripts directory..."

# Get the actual username (not root)
ACTUAL_USER="${SUDO_USER:-$USER}"
USER_HOME=$(eval echo ~$ACTUAL_USER)

mkdir -p "$USER_HOME/scripts"
mkdir -p "$USER_HOME/data"

# -----------------------------------------------------------------------------
# Step 3: Copy the backup script
# -----------------------------------------------------------------------------
echo ""
echo "[3/6] Installing backup script..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -f "$SCRIPT_DIR/scripts/zip.sh" ]; then
    cp "$SCRIPT_DIR/scripts/zip.sh" "$USER_HOME/scripts/zip.sh"
else
    cat > "$USER_HOME/scripts/zip.sh" << SCRIPT
#!/bin/bash

SOURCE_DIR="$USER_HOME/data"
ZIP_NAME="data_backup"
LOCAL_OUTPUT_DIR="$USER_HOME"
REMOTE_USER="jason"
REMOTE_HOST="fedora.local"
REMOTE_PATH="/home/jason/backups"
MAX_BACKUPS=3      # Keep only the last 3 local backups

set -e

TIMESTAMP=\$(date +"%Y-%m-%d_%H-%M-%S")
LOG_FILE="$USER_HOME/backup.log"

echo "$(date): Starting backup..." >> "$LOG_FILE"

if [ ! -d "$SOURCE_DIR" ]; then
    echo "ERROR: Source directory missing!" >> "$LOG_FILE"
    exit 1
fi

ZIP_FILE="$LOCAL_OUTPUT_DIR/${ZIP_NAME}_${TIMESTAMP}.zip"

echo "Creating zip: \$ZIP_FILE" >> "\$LOG_FILE"
cd "\$SOURCE_DIR"
zip -r "\$ZIP_FILE" ./* >> "\$LOG_FILE" 2>&1

# Ensure user owns it
chown $ACTUAL_USER:$ACTUAL_USER "\$ZIP_FILE"

echo "Creating remote directory..." >> "$LOG_FILE"
ssh "$REMOTE_USER@$REMOTE_HOST" "mkdir -p $REMOTE_PATH"

echo "Copying zip to remote..." >> "$LOG_FILE"
scp "$ZIP_FILE" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH"

echo "Backup complete: $ZIP_FILE" >> "$LOG_FILE"

# ===== CLEANUP OLD LOCAL BACKUPS =====

echo "Cleaning up old local backups..." >> "$LOG_FILE"

cd "$LOCAL_OUTPUT_DIR"

# List zip files newest → oldest
BACKUPS=( $(ls -1t ${ZIP_NAME}_*.zip 2>/dev/null) )

COUNT=${#BACKUPS[@]}

if (( COUNT > MAX_BACKUPS )); then
    for (( i=MAX_BACKUPS; i<COUNT; i++ )); do
        echo "Deleting old backup: ${BACKUPS[$i]}" >> "$LOG_FILE"
        rm -f "${BACKUPS[$i]}"
    done
else
    echo "No cleanup needed; $COUNT backups present" >> "$LOG_FILE"
fi

echo "Backup + cleanup finished successfully." >> "$LOG_FILE"
sudo shutdown now
SCRIPT
fi

chmod +x "$USER_HOME/scripts/zip.sh"
chown "$ACTUAL_USER:$ACTUAL_USER" "$USER_HOME/scripts/zip.sh"

# -----------------------------------------------------------------------------
# Step 4: Configure logind.conf to ignore power button
# -----------------------------------------------------------------------------
echo ""
echo "[4/6] Configuring systemd logind..."

# Backup original config
cp /etc/systemd/logind.conf /etc/systemd/logind.conf.backup

# Update or add HandlePowerKey settings
if grep -q "^#HandlePowerKey=" /etc/systemd/logind.conf; then
    sed -i 's/^#HandlePowerKey=.*/HandlePowerKey=ignore/' /etc/systemd/logind.conf
elif grep -q "^HandlePowerKey=" /etc/systemd/logind.conf; then
    sed -i 's/^HandlePowerKey=.*/HandlePowerKey=ignore/' /etc/systemd/logind.conf
else
    echo "HandlePowerKey=ignore" >> /etc/systemd/logind.conf
fi

if grep -q "^#HandlePowerKeyLongPress=" /etc/systemd/logind.conf; then
    sed -i 's/^#HandlePowerKeyLongPress=.*/HandlePowerKeyLongPress=ignore/' /etc/systemd/logind.conf
elif grep -q "^HandlePowerKeyLongPress=" /etc/systemd/logind.conf; then
    sed -i 's/^HandlePowerKeyLongPress=.*/HandlePowerKeyLongPress=ignore/' /etc/systemd/logind.conf
else
    echo "HandlePowerKeyLongPress=ignore" >> /etc/systemd/logind.conf
fi

# -----------------------------------------------------------------------------
# Step 5: Configure triggerhappy
# -----------------------------------------------------------------------------
echo ""
echo "[5/6] Configuring triggerhappy..."

cat > /etc/triggerhappy/triggers.d/power-backup.conf << TRIGGER
KEY_POWER 1 sudo -u $ACTUAL_USER $USER_HOME/scripts/zip.sh
TRIGGER

# Restart triggerhappy service
systemctl restart triggerhappy

# -----------------------------------------------------------------------------
# Step 6: Configure sudoers for passwordless execution
# -----------------------------------------------------------------------------
echo ""
echo "[6/6] Configuring sudo permissions..."

# Add sudoers entry for triggerhappy (thd runs as nobody)
SUDOERS_LINE="nobody ALL=($ACTUAL_USER) NOPASSWD: $USER_HOME/scripts/zip.sh"

if ! grep -q "$SUDOERS_LINE" /etc/sudoers.d/power-backup 2>/dev/null; then
    echo "$SUDOERS_LINE" > /etc/sudoers.d/power-backup
    chmod 440 /etc/sudoers.d/power-backup
fi

# Also allow user to shutdown without password
USER_SHUTDOWN="$ACTUAL_USER ALL=(ALL) NOPASSWD: /sbin/shutdown"
if ! grep -q "$USER_SHUTDOWN" /etc/sudoers.d/power-backup 2>/dev/null; then
    echo "$USER_SHUTDOWN" >> /etc/sudoers.d/power-backup
fi

# -----------------------------------------------------------------------------
# Done!
# -----------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "IMPORTANT: You still need to set up SSH keys for remote backup:"
echo ""
echo "  1. Generate SSH key (as $ACTUAL_USER user):"
echo "     ssh-keygen -t ed25519"
echo ""
echo "  2. Copy key to remote server:"
echo "     ssh-copy-id ${REMOTE_USER:-jason}@${REMOTE_HOST:-fedora.local}"
echo ""
echo "  3. Test the connection:"
echo "     ssh ${REMOTE_USER:-jason}@${REMOTE_HOST:-fedora.local}"
echo ""
echo "  4. Make sure the data directory exists:"
echo "     mkdir -p $USER_HOME/data"
echo ""
echo "  5. Reboot to apply logind changes:"
echo "     sudo reboot"
echo ""
echo "After reboot, pressing the power button will:"
echo "  - Backup $USER_HOME/data to a zip file"
echo "  - Copy the backup to the remote server"
echo "  - Clean up old local backups (keeping last 3)"
echo "  - Shutdown the Pi"
echo ""
