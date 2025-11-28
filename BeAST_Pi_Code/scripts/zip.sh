#!/bin/bash

SOURCE_DIR="/home/pi/data"
ZIP_NAME="data_backup"
LOCAL_OUTPUT_DIR="/home/pi"
REMOTE_USER="jason"
REMOTE_HOST="fedora.local"
REMOTE_PATH="/home/jason/backups"
MAX_BACKUPS=3      # Keep only the last 3 local backups

set -e

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_FILE="/home/pi/backup.log"

echo "$(date): Starting backup..." >> "$LOG_FILE"

if [ ! -d "$SOURCE_DIR" ]; then
    echo "ERROR: Source directory missing!" >> "$LOG_FILE"
    exit 1
fi

ZIP_FILE="$LOCAL_OUTPUT_DIR/${ZIP_NAME}_${TIMESTAMP}.zip"

echo "Creating zip: $ZIP_FILE" >> "$LOG_FILE"
cd "$SOURCE_DIR"
zip -r "$ZIP_FILE" ./* >> "$LOG_FILE" 2>&1

# Ensure pi owns it
chown pi:pi "$ZIP_FILE"

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
