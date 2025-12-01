# BeAST - Biometric Enhancement & Adaptive Sensing Technology

A wearable AI system for the Raspberry Pi that collects and analyzes biometric data from Arduino sensors.

## Project Structure

```
Beast/
├── BeAST_Pi_Code/          # Main Raspberry Pi codebase
│   ├── beast_session_runner.sh    # Session runner with automatic venv setup
│   ├── BeAST Schema/              # Database schemas and synthetic data
│   ├── Live Connections Simulator Test/  # Arduino-to-SQL data collection
│   ├── TheBeast/                  # Voice assistant and RAG system
│   └── scripts/                   # Utility scripts (backup, etc.)
```

## Key Features

- **Arduino Data Collection**: Real-time EEG and biometric data streaming to PostgreSQL
- **Voice Assistant**: AI-powered voice interface with local models
- **Database Management**: PostgreSQL with comprehensive schema for sensor data
- **Automated Backups**: Power button integration for safe shutdown with data backup
- **Service Management**: Systemd services for automatic startup

## Installation

All scripts automatically detect the current user and use relative paths. No hardcoding required!

### 1. Database Setup
The database schema is located in `BeAST Schema/beast_schema.sql`

### 2. Arduino Data Collector
Run the session runner:
```bash
./beast_session_runner.sh
```

This script automatically:
- Detects the current user
- Creates a virtual environment if needed
- Installs dependencies
- Runs the Arduino-to-SQL data collector

### 3. Voice Assistant Setup
Navigate to `TheBeast/TheBeast/beast/` and run:
```bash
python install.py
```

### 4. Power Button Backup (Optional)
Configure automatic backup on power button press:
```bash
sudo ./setup_power_button.sh
```

### 5. Remote Backup Server (Optional)
On the receiving computer:
```bash
sudo ./setup_big_database_computer.sh
```

## Service Files

The project includes systemd service templates that use systemd specifiers for portability:
- `%u` = current username
- `%h` = user's home directory

These services will work on any system without modification:
- `beast-arduino.service` - Arduino data collector
- `beast-voice.service` - Voice assistant
- `beast.service` - Main BeAST service (template version)
- `beast-autostart.service` - Auto-start variant

## Configuration

All configuration is user-agnostic and uses environment detection:
- The session runner automatically finds the project directory
- Service files use systemd specifiers
- Setup scripts detect the current user automatically

## Development

### Data Formats
See `Live Connections Simulator Test/DATA_FORMATS.md` for sensor data formats.

### Developer Setup
See `Live Connections Simulator Test/DEVELOPER_SETUP.md` for development environment setup.

## Notes

- The system was designed to run on multiple Raspberry Pi units (beast1, beast2, beast3, beast4, etc.)
- All scripts now auto-detect the current user and adapt paths accordingly
- No hardcoded usernames or paths in the main codebase
