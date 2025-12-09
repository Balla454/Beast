# BeAST Installation Guide

## Fresh Installation on Any Raspberry Pi

### Prerequisites
- Raspberry Pi 4/5 with Raspberry Pi OS
- Internet connection (for initial pip installs)
- Audio output device
- Python 3.7+ (pre-installed on Raspberry Pi OS)

### Quick Install

1. **Clone or Transfer the Repository**
   ```bash
   # Option A: Clone from GitHub
   cd ~
   git clone https://github.com/Balla454/Beast.git
   
   # Option B: Extract from zip file
   cd ~
   unzip Beast.zip
   ```

2. **Run the Setup Script**
   ```bash
   cd ~/Beast/BeAST_Pi_Code
   ./install.sh
   ```
   
   This script will:
   - Install system dependencies (apt)
   - Create a fresh Python virtual environment
   - Install all Python dependencies
   - Download necessary AI models
   - Configure user permissions
   - Install and start system services

### What Gets Installed

**Services:**
- `beast-voice.service` - Main voice assistant
- `beast-synthetic.service` - Synthetic data playback (for testing)

**Virtual Environment:**
- Located at `~/.beast-venv/`
- Automatically recreated on each machine
- **Note:** Virtual environments are NOT portable between computers

### Manual Installation

If you prefer to install manually:

```bash
# 1. Create virtual environment
python3 -m venv ~/.beast-venv
source ~/.beast-venv/bin/activate

# 2. Install dependencies
pip install -r ~/Beast/BeAST_Pi_Code/TheBeast/TheBeast/beast/requirements.txt

# 3. Add user to audio group
sudo usermod -aG audio $USER

# 4. Install services
sudo cp ~/Beast/BeAST_Pi_Code/beast-voice.service /etc/systemd/system/
sudo cp ~/Beast/BeAST_Pi_Code/beast-synthetic.service /etc/systemd/system/

# 5. Enable and start services
sudo systemctl daemon-reload
sudo systemctl enable beast-voice.service beast-synthetic.service
sudo systemctl start beast-voice.service beast-synthetic.service
```

### Verify Installation

```bash
# Check service status
systemctl status beast-voice.service

# View live logs
journalctl -u beast-voice.service -f
```

### Troubleshooting

**Issue:** Service fails to start
```bash
# Check logs for errors
journalctl -u beast-voice.service -n 50

# Verify virtual environment exists
ls -la ~/.beast-venv/

# Manually recreate if needed
rm -rf ~/.beast-venv
python3 -m venv ~/.beast-venv
source ~/.beast-venv/bin/activate
pip install -r ~/Beast/BeAST_Pi_Code/TheBeast/TheBeast/beast/requirements.txt
```

**Issue:** Permission denied
```bash
# Fix file ownership
sudo chown -R $USER:$USER ~/Beast

# Re-add to audio group (requires logout/login)
sudo usermod -aG audio $USER
```

**Issue:** Database errors
```bash
# For PostgreSQL setup (optional)
sudo apt install postgresql postgresql-contrib
sudo -u postgres psql -c "CREATE DATABASE beast;"
sudo -u postgres psql -c "CREATE USER beast WITH PASSWORD 'beast';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE beast TO beast;"
```

### Transferring Between Computers

When moving BeAST to a new computer:

1. **DO NOT** copy the `.beast-venv/` directory (it won't work)
2. Copy the entire `Beast/` folder (excluding `.beast-venv/`)
3. Run `setup_services.sh` on the new machine
4. The script will automatically create a fresh virtual environment

### Uninstalling

```bash
# Stop and disable services
sudo systemctl stop beast-voice.service beast-synthetic.service
sudo systemctl disable beast-voice.service beast-synthetic.service

# Remove service files
sudo rm /etc/systemd/system/beast-voice.service
sudo rm /etc/systemd/system/beast-synthetic.service
sudo systemctl daemon-reload

# Remove virtual environment
rm -rf ~/.beast-venv

# Remove Beast directory (optional)
rm -rf ~/Beast
```

## Additional Resources

- [GitHub Repository](https://github.com/Balla454/Beast)
- Service files: `BeAST_Pi_Code/*.service`
- Setup script: `BeAST_Pi_Code/setup_services.sh`
