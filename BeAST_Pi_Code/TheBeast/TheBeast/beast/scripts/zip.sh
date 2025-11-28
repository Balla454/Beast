#!/bin/bash
# =============================================================================
# BeAST Data Sync Script
# =============================================================================
# Triggered by power button via triggerhappy
# 1. Creates timestamped backup of data directory
# 2. Syncs to host PC via SCP
# 3. Cleans up old local backups
# 4. Shuts down the system
# =============================================================================

set -e

# Configuration via environment variables with sensible defaults
BEAST_HOME="${BEAST_HOME:-$(dirname "$(dirname "$(readlink -f "$0")")")}"
DATA_DIR="${BEAST_DATA_DIR:-$BEAST_HOME/data}"
BACKUP_DIR="${BEAST_BACKUP_DIR:-$BEAST_HOME/backups}"
HOST_PC="${BEAST_BACKUP_HOST:-}"
HOST_USER="${BEAST_BACKUP_USER:-$(whoami)}"
HOST_PATH="${BEAST_BACKUP_PATH:-/home/${HOST_USER}/beast_backups}"
MAX_LOCAL_BACKUPS="${BEAST_MAX_BACKUPS:-5}"
LOG_FILE="${BEAST_LOG_DIR:-$BEAST_HOME/logs}/sync.log"
PIPER_MODEL="${PIPER_MODEL_PATH:-$BEAST_HOME/models/tts/en_US-lessac-medium.onnx}"

# Ensure log directory exists
mkdir -p "$(dirname "$LOG_FILE")"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Speak announcement if TTS available
announce() {
    if command -v espeak &> /dev/null; then
        espeak -v en-us -s 150 "$1" 2>/dev/null &
    elif command -v piper &> /dev/null && [ -f "$PIPER_MODEL" ]; then
        echo "$1" | piper --model "$PIPER_MODEL" --output-raw | aplay -q 2>/dev/null &
    fi
}

# Main sync function
main() {
    log "=========================================="
    log "BeAST Data Sync Started"
    log "BEAST_HOME: $BEAST_HOME"
    log "DATA_DIR: $DATA_DIR"
    log "=========================================="
    
    # Announce sync starting
    announce "Syncing data to host computer"
    
    # Create backup directory
    mkdir -p "$BACKUP_DIR"
    
    # Generate timestamp
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    BACKUP_NAME="beast_backup_${TIMESTAMP}.tar.gz"
    BACKUP_PATH="${BACKUP_DIR}/${BACKUP_NAME}"
    
    # Create compressed backup
    log "Creating backup: $BACKUP_NAME"
    
    if [ -d "$DATA_DIR" ] && [ "$(ls -A $DATA_DIR 2>/dev/null)" ]; then
        tar -czf "$BACKUP_PATH" -C "$(dirname $DATA_DIR)" "$(basename $DATA_DIR)" 2>/dev/null
        
        BACKUP_SIZE=$(du -h "$BACKUP_PATH" | cut -f1)
        log "Backup created: $BACKUP_SIZE"
    else
        log "No data to backup in $DATA_DIR"
        # Create empty marker file
        touch "$BACKUP_PATH.empty"
        BACKUP_PATH="$BACKUP_PATH.empty"
    fi
    
    # Attempt to sync to host PC (only if HOST_PC is configured)
    SYNC_SUCCESS=false
    
    if [ -n "$HOST_PC" ]; then
        log "Attempting to sync to ${HOST_PC}..."
        
        # Check if host is reachable
        if ping -c 1 -W 2 "$HOST_PC" &> /dev/null; then
            log "Host reachable, starting SCP transfer..."
            
            # Create remote directory if needed
            ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no \
                "${HOST_USER}@${HOST_PC}" "mkdir -p ${HOST_PATH}" 2>/dev/null || true
            
            # Transfer backup
            if scp -o ConnectTimeout=10 -o StrictHostKeyChecking=no \
                   "$BACKUP_PATH" "${HOST_USER}@${HOST_PC}:${HOST_PATH}/" 2>/dev/null; then
                log "Sync successful: ${HOST_PC}:${HOST_PATH}/${BACKUP_NAME}"
                SYNC_SUCCESS=true
                announce "Data sync complete"
            else
                log "SCP transfer failed"
            fi
        else
            log "Host ${HOST_PC} not reachable"
        fi
        
        if [ "$SYNC_SUCCESS" = false ]; then
            log "WARNING: Remote sync failed, backup preserved locally"
            announce "Sync failed. Data saved locally"
        fi
    else
        log "No remote host configured (set BEAST_BACKUP_HOST)"
        log "Backup preserved locally at: $BACKUP_PATH"
    fi
    
    # Clean up old local backups (keep last N)
    log "Cleaning old backups (keeping last $MAX_LOCAL_BACKUPS)..."
    
    BACKUP_COUNT=$(ls -1 "$BACKUP_DIR"/beast_backup_*.tar.gz 2>/dev/null | wc -l)
    
    if [ "$BACKUP_COUNT" -gt "$MAX_LOCAL_BACKUPS" ]; then
        DELETE_COUNT=$((BACKUP_COUNT - MAX_LOCAL_BACKUPS))
        ls -1t "$BACKUP_DIR"/beast_backup_*.tar.gz 2>/dev/null | tail -n "$DELETE_COUNT" | xargs rm -f
        log "Deleted $DELETE_COUNT old backup(s)"
    fi
    
    # Log completion
    log "Sync process completed"
    log "=========================================="
    
    # Announce shutdown
    announce "Shutting down. Goodbye"
    sleep 2  # Allow announcement to complete
    
    # Shutdown the system
    log "Initiating system shutdown..."
    sudo shutdown -h now
}

# Run main function
main "$@"
